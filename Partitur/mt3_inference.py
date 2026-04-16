"""
MT3 Inference Wrapper for Partitur.

This module provides the InferenceModel class that uses a pre-trained MT3
(Multi-Task Multitrack Music Transcription) Transformer model to transcribe
audio into MIDI note sequences — replacing the old FFT-based approach.

Based on the official MT3 Colab notebook:
https://colab.research.google.com/github/magenta/mt3/blob/main/mt3/colab/music_transcription_with_transformers.ipynb
"""

import functools
import os
import sys

import numpy as np
import tensorflow.compat.v2 as tf

# ---------------------------------------------------------------------------
# Resolve paths relative to this file so imports work from any working dir.
# ---------------------------------------------------------------------------
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_THIS_DIR)          # IntoSheet/
_MT3_ROOT = os.path.join(_PROJECT_ROOT, "mt3")       # IntoSheet/mt3/
_T5X_ROOT = os.path.join(_PROJECT_ROOT, "t5x")       # IntoSheet/t5x/

# Make sure the mt3 package is importable.
if _MT3_ROOT not in sys.path:
    sys.path.insert(0, _MT3_ROOT)
if _T5X_ROOT not in sys.path:
    sys.path.insert(0, _T5X_ROOT)

import gin
import jax
import librosa
import note_seq
import seqio
import t5
import t5x

from mt3 import metrics_utils
from mt3 import models
from mt3 import network
from mt3 import note_sequences
from mt3 import preprocessors
from mt3 import spectrograms
from mt3 import vocabularies

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
SAMPLE_RATE = 16000
CHECKPOINT_PATH = os.path.join(_PROJECT_ROOT, "checkpoints", "mt3")
GIN_DIR = os.path.join(_MT3_ROOT, "mt3", "gin")

# MIDI note number → note name look-up (scientific pitch notation)
_NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]


def midi_pitch_to_name(pitch: int) -> str:
    """Convert a MIDI pitch number (0-127) to a human-readable note name."""
    octave = (pitch // 12) - 1
    note = _NOTE_NAMES[pitch % 12]
    return f"{note}{octave}"


# ---------------------------------------------------------------------------
# InferenceModel
# ---------------------------------------------------------------------------
class InferenceModel:
    """Wrapper around a T5X model for music transcription.

    Usage::

        model = InferenceModel()          # loads checkpoint
        ns = model.transcribe("song.wav") # returns note_seq.NoteSequence
    """

    def __init__(
        self,
        checkpoint_path: str = CHECKPOINT_PATH,
        model_type: str = "mt3",
    ):
        # ----- model constants ------------------------------------------------
        if model_type == "ismir2021":
            num_velocity_bins = 127
            self.encoding_spec = note_sequences.NoteEncodingSpec
            self.inputs_length = 512
        elif model_type == "mt3":
            num_velocity_bins = 1
            self.encoding_spec = note_sequences.NoteEncodingWithTiesSpec
            self.inputs_length = 256
        else:
            raise ValueError(f"unknown model_type: {model_type}")

        gin_files = [
            os.path.join(GIN_DIR, "model.gin"),
            os.path.join(GIN_DIR, f"{model_type}.gin"),
        ]

        self.batch_size = 8
        self.outputs_length = 1024
        self.sequence_length = {
            "inputs": self.inputs_length,
            "targets": self.outputs_length,
        }

        self.partitioner = t5x.partitioning.PjitPartitioner(num_partitions=1)

        # ----- codec / vocabulary ---------------------------------------------
        self.spectrogram_config = spectrograms.SpectrogramConfig()
        self.codec = vocabularies.build_codec(
            vocab_config=vocabularies.VocabularyConfig(
                num_velocity_bins=num_velocity_bins
            )
        )
        self.vocabulary = vocabularies.vocabulary_from_codec(self.codec)
        self.output_features = {
            "inputs": seqio.ContinuousFeature(dtype=tf.float32, rank=2),
            "targets": seqio.Feature(vocabulary=self.vocabulary),
        }

        # ----- T5X model ------------------------------------------------------
        self._parse_gin(gin_files)
        self.model = self._load_model()
        self.restore_from_checkpoint(checkpoint_path)

    # ------------------------------------------------------------------
    # internal helpers
    # ------------------------------------------------------------------
    @property
    def input_shapes(self):
        return {
            "encoder_input_tokens": (self.batch_size, self.inputs_length),
            "decoder_input_tokens": (self.batch_size, self.outputs_length),
        }

    def _parse_gin(self, gin_files):
        gin_bindings = [
            "from __gin__ import dynamic_registration",
            "from mt3 import vocabularies",
            "VOCAB_CONFIG=@vocabularies.VocabularyConfig()",
            "vocabularies.VocabularyConfig.num_velocity_bins=%NUM_VELOCITY_BINS",
        ]
        with gin.unlock_config():
            gin.parse_config_files_and_bindings(
                gin_files, gin_bindings, finalize_config=False
            )

    def _load_model(self):
        model_config = gin.get_configurable(network.T5Config)()
        module = network.Transformer(config=model_config)
        return models.ContinuousInputsEncoderDecoderModel(
            module=module,
            input_vocabulary=self.output_features["inputs"].vocabulary,
            output_vocabulary=self.output_features["targets"].vocabulary,
            optimizer_def=t5x.adafactor.Adafactor(decay_rate=0.8, step_offset=0),
            input_depth=spectrograms.input_depth(self.spectrogram_config),
        )

    def restore_from_checkpoint(self, checkpoint_path):
        train_state_initializer = t5x.utils.TrainStateInitializer(
            optimizer_def=self.model.optimizer_def,
            init_fn=self.model.get_initial_variables,
            input_shapes=self.input_shapes,
            partitioner=self.partitioner,
        )
        restore_checkpoint_cfg = t5x.utils.RestoreCheckpointConfig(
            path=checkpoint_path, mode="specific", dtype="float32"
        )
        train_state_axes = train_state_initializer.train_state_axes
        self._predict_fn = self._get_predict_fn(train_state_axes)
        self._train_state = train_state_initializer.from_checkpoint_or_scratch(
            [restore_checkpoint_cfg], init_rng=jax.random.PRNGKey(0)
        )

    @functools.lru_cache()
    def _get_predict_fn(self, train_state_axes):
        def partial_predict_fn(params, batch, decode_rng):
            return self.model.predict_batch_with_aux(
                params, batch, decoder_params={"decode_rng": None}
            )

        return self.partitioner.partition(
            partial_predict_fn,
            in_axis_resources=(
                train_state_axes.params,
                t5x.partitioning.PartitionSpec("data"),
                None,
            ),
            out_axis_resources=t5x.partitioning.PartitionSpec("data"),
        )

    def predict_tokens(self, batch, seed=0):
        prediction, _ = self._predict_fn(
            self._train_state.params, batch, jax.random.PRNGKey(seed)
        )
        return self.vocabulary.decode_tf(prediction).numpy()

    # ------------------------------------------------------------------
    # audio → spectrogram helpers
    # ------------------------------------------------------------------
    def _audio_to_frames(self, audio):
        frame_size = self.spectrogram_config.hop_width
        padding = [0, frame_size - len(audio) % frame_size]
        audio = np.pad(audio, padding, mode="constant")
        frames = spectrograms.split_audio(audio, self.spectrogram_config)
        num_frames = len(audio) // frame_size
        times = np.arange(num_frames) / self.spectrogram_config.frames_per_second
        return frames, times

    def audio_to_dataset(self, audio):
        frames, frame_times = self._audio_to_frames(audio)
        return tf.data.Dataset.from_tensors(
            {"inputs": frames, "input_times": frame_times}
        )

    def preprocess(self, ds):
        pp_chain = [
            functools.partial(
                t5.data.preprocessors.split_tokens_to_inputs_length,
                sequence_length=self.sequence_length,
                output_features=self.output_features,
                feature_key="inputs",
                additional_feature_keys=["input_times"],
            ),
            preprocessors.add_dummy_targets,
            functools.partial(
                preprocessors.compute_spectrograms,
                spectrogram_config=self.spectrogram_config,
            ),
        ]
        for pp in pp_chain:
            ds = pp(ds)
        return ds

    def postprocess(self, tokens, example):
        tokens = self._trim_eos(tokens)
        start_time = example["input_times"][0]
        start_time -= start_time % (1 / self.codec.steps_per_second)
        return {
            "est_tokens": tokens,
            "start_time": start_time,
            "raw_inputs": [],
        }

    @staticmethod
    def _trim_eos(tokens):
        tokens = np.array(tokens, np.int32)
        if vocabularies.DECODED_EOS_ID in tokens:
            tokens = tokens[: np.argmax(tokens == vocabularies.DECODED_EOS_ID)]
        return tokens

    # ------------------------------------------------------------------
    # public API
    # ------------------------------------------------------------------
    def transcribe(self, audio_path: str) -> "note_seq.NoteSequence":
        """Transcribe an audio file and return a ``NoteSequence``.

        Args:
            audio_path: Path to a WAV or MP3 file.

        Returns:
            A ``note_seq.NoteSequence`` containing the transcribed notes.
        """
        audio, _ = librosa.load(audio_path, sr=SAMPLE_RATE, mono=True)
        return self._transcribe_audio(audio)

    def _transcribe_audio(self, audio: np.ndarray) -> "note_seq.NoteSequence":
        """Core transcription from a 1-D numpy array of 16 kHz samples."""
        ds = self.audio_to_dataset(audio)
        ds = self.preprocess(ds)

        model_ds = self.model.FEATURE_CONVERTER_CLS(pack=False)(
            ds, task_feature_lengths=self.sequence_length
        )
        model_ds = model_ds.batch(self.batch_size)

        inferences = (
            tokens
            for batch in model_ds.as_numpy_iterator()
            for tokens in self.predict_tokens(batch)
        )

        predictions = []
        for example, tokens in zip(ds.as_numpy_iterator(), inferences):
            predictions.append(self.postprocess(tokens, example))

        result = metrics_utils.event_predictions_to_ns(
            predictions, codec=self.codec, encoding_spec=self.encoding_spec
        )
        return result["est_ns"]

    # ------------------------------------------------------------------
    # convenience: extract note list in Partitur-compatible format
    # ------------------------------------------------------------------
    def transcribe_to_notes(self, audio_path: str):
        """Transcribe audio and return notes as a list of dicts.

        Each dict contains:
            - ``name``    : note name, e.g. ``"C4"``
            - ``pitch``   : MIDI pitch number (0-127)
            - ``start``   : start time in seconds
            - ``end``     : end time in seconds
            - ``velocity``: MIDI velocity (0-127)
            - ``program`` : MIDI program number
            - ``is_drum`` : whether the note is a drum hit

        Returns:
            A list of note dicts sorted by start time.
        """
        ns = self.transcribe(audio_path)
        notes = []
        for n in ns.notes:
            notes.append(
                {
                    "name": midi_pitch_to_name(n.pitch),
                    "pitch": n.pitch,
                    "start": round(n.start_time, 4),
                    "end": round(n.end_time, 4),
                    "velocity": n.velocity,
                    "program": n.program,
                    "is_drum": n.is_drum,
                }
            )
        notes.sort(key=lambda x: x["start"])
        return notes

    def transcribe_to_midi(self, audio_path: str, midi_path: str):
        """Transcribe audio and save the result as a MIDI file."""
        ns = self.transcribe(audio_path)
        note_seq.sequence_proto_to_midi_file(ns, midi_path)
        print(f"MIDI saved to: {midi_path}")
        return ns
