#!/bin/bash
#
echo "========================================"
echo "  IntoSheet — MT3 기반 음악 채보 시스템  "
echo "========================================"

# ──────────────────────────────────────────────
# 경로 설정 (필요시 수정)
# ──────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
MT3_DIR="${SCRIPT_DIR}/mt3"
T5X_DIR="${SCRIPT_DIR}/t5x"
CHECKPOINT_DIR="${SCRIPT_DIR}/checkpoints/mt3"
MODEL_DIR="${SCRIPT_DIR}/training_output"      # 학습 결과 저장 경로
GIN_DIR="${MT3_DIR}/mt3/gin"

# ──────────────────────────────────────────────
# 사용법
# ──────────────────────────────────────────────
usage() {
    echo ""
    echo "사용법:"
    echo "  bash run_all.sh train [옵션]         MT3 모델 학습 (fine-tuning)"
    echo "  bash run_all.sh infer <audio_file>   오디오 파일 채보 (추론)"
    echo "  bash run_all.sh pdf <midi_file>      MIDI를 PDF 악보로 변환"
    echo "  bash run_all.sh install              의존성 패키지 설치"
    echo ""
    echo "학습 옵션:"
    echo "  --model_type=mt3|ismir2021    모델 유형 (기본: mt3)"
    echo "  --steps=N                     학습 스텝 수 (기본: 1000000)"
    echo "  --batch_size=N                배치 크기 (기본: 256)"
    echo "  --use_cached_tasks=true|false SeqIO 캐시 사용 여부 (기본: false)"
    echo "  --model_dir=PATH              체크포인트 저장 경로"
    echo "  --init_checkpoint=PATH        초기 체크포인트 경로 (fine-tuning 시)"
    echo "  --tfds_data_dir=PATH          TFDS 데이터 디렉토리"
    echo ""
    echo "예시:"
    echo "  bash run_all.sh install"
    echo "  python -m pip install -e ."
    echo "  bash run_all.sh train --model_type=mt3 --steps=10000"
    echo "  bash run_all.sh train --init_checkpoint=${CHECKPOINT_DIR}"
    echo "  bash run_all.sh infer /path/to/audio.wav"
    echo "  bash run_all.sh pdf ./output/song.mid"
    echo "  (conda activate mt3_env)  bash run_all.sh train --init_checkpoint=checkpoints/mt3 --steps=10000 --batch_size=4 --use_cached_tasks=false"
}

# ──────────────────────────────────────────────
# install: 의존성 설치
# ──────────────────────────────────────────────
do_install() {
    if ! python -c "import sys; raise SystemExit(0 if sys.version_info[:2] == (3, 11) else 1)"; then
        echo "❌ 오류: Python 3.11 환경에서 설치를 진행해야 합니다."
        echo "   현재 버전: $(python --version 2>&1)"
        return 1
    fi

    echo "[1/4] Python 3.11 환경을 확인했습니다..."
    python --version || return 1

    echo "[2/4] pip를 최신 상태로 맞춥니다..."
    python -m pip install --upgrade pip || return 1

    echo "[3/4] 루트 setup.py 기준으로 IntoSheet 전체를 설치합니다..."
    python -m pip install -e "${SCRIPT_DIR}" --no-build-isolation || return 1

    echo "[4/4] PDF 출력 디렉터리를 준비합니다..."
    mkdir -p "${SCRIPT_DIR}/checkpoints/pdf_output" || return 1

    echo ""
    echo "✅ 설치 완료!"
}

# ──────────────────────────────────────────────
# train: MT3 모델 학습
# ──────────────────────────────────────────────
do_train() {
    # 기본값
    local model_type="mt3"
    local steps=""
    local batch_size=""
    local use_cached_tasks="false"
    local model_dir="${MODEL_DIR}"
    local init_checkpoint=""
    local tfds_data_dir=""

    # 인자 파싱
    for arg in "$@"; do
        case $arg in
            --model_type=*)   model_type="${arg#*=}" ;;
            --steps=*)        steps="${arg#*=}" ;;
            --batch_size=*)   batch_size="${arg#*=}" ;;
            --use_cached_tasks=*) use_cached_tasks="${arg#*=}" ;;
            --model_dir=*)    model_dir="${arg#*=}" ;;
            --init_checkpoint=*) init_checkpoint="${arg#*=}" ;;
            --tfds_data_dir=*) tfds_data_dir="${arg#*=}" ;;
            *) echo "알 수 없는 옵션: $arg"; usage; exit 1 ;;
        esac
    done

    if [[ "${use_cached_tasks}" != "true" && "${use_cached_tasks}" != "false" ]]; then
        echo "❌ 오류: --use_cached_tasks 는 true 또는 false 여야 합니다. (입력값: ${use_cached_tasks})"
        exit 1
    fi
    local use_cached_tasks_gin="False"
    if [[ "${use_cached_tasks}" == "true" ]]; then
        use_cached_tasks_gin="True"
    fi

    # 모델 유형에 따른 gin 파일 선택
    local model_gin="${GIN_DIR}/${model_type}.gin"
    if [ ! -f "$model_gin" ]; then
        echo "❌ 오류: gin 파일을 찾을 수 없습니다: ${model_gin}"
        echo "   사용 가능한 모델 유형: mt3, ismir2021"
        exit 1
    fi

    echo "========================================"
    echo "  MT3 학습 시작"
    echo "  모델 유형:    ${model_type}"
    echo "  모델 저장:    ${model_dir}"
    if [ -n "$init_checkpoint" ]; then
        echo "  초기 체크포인트: ${init_checkpoint}"
    fi
    echo "========================================"

    mkdir -p "${model_dir}"

    # PYTHONPATH에 mt3 추가 (tasks.py 등을 gin에서 import 하기 위해)
    export PYTHONPATH="${MT3_DIR}:${PYTHONPATH}"

    # 명령어를 배열로 구성 (괄호 등 특수문자 보호)
    local -a CMD_ARGS=(
        python "${T5X_DIR}/t5x/train.py"
        "--gin_search_paths=${GIN_DIR}"
        "--gin_file=${GIN_DIR}/model.gin"
        "--gin_file=${GIN_DIR}/train.gin"
        "--gin_file=${model_gin}"
        "--gin.USE_CACHED_TASKS=${use_cached_tasks_gin}"
        "--gin.MODEL_DIR='${model_dir}'"
    )

    if [ -n "$steps" ]; then
        CMD_ARGS+=("--gin.TRAIN_STEPS=${steps}")
    fi

    if [ -n "$batch_size" ]; then
        CMD_ARGS+=("--gin.BATCH_SIZE=${batch_size}")
    fi

    if [ -n "$init_checkpoint" ]; then
        CMD_ARGS+=("--gin.utils.CheckpointConfig.restore=@utils.RestoreCheckpointConfig()")
        CMD_ARGS+=("--gin.utils.RestoreCheckpointConfig.path='${init_checkpoint}'")
        CMD_ARGS+=("--gin.utils.RestoreCheckpointConfig.mode='specific'")
        CMD_ARGS+=("--gin.utils.RestoreCheckpointConfig.dtype='float32'")
    fi

    if [ -n "$tfds_data_dir" ]; then
        CMD_ARGS+=("--tfds_data_dir=${tfds_data_dir}")
    fi

    echo ""
    echo "실행 명령어:"
    echo "  ${CMD_ARGS[*]}"
    echo ""

    "${CMD_ARGS[@]}"
}

# ──────────────────────────────────────────────
# infer: 오디오 채보 (추론)
# ──────────────────────────────────────────────
do_infer() {
    local audio_file="$1"

    if [ -z "$audio_file" ]; then
        echo "----------------------------------------"
        echo "주의: 파일 입력시 .mp3 또는 .wav의 경로를 입력해주세요."
        echo "      예: /media/lucius/SATA_SSD/IntoSheet/test_audio.wav"
        echo "      커맨드라인 인자로도 전달 가능합니다:"
        echo "      예: bash run_all.sh infer /path/to/audio.wav"
        echo "----------------------------------------"
    fi

    python "${SCRIPT_DIR}/Partitur/run.py" ${audio_file}
}

# ──────────────────────────────────────────────
# pdf: MIDI -> PDF 악보 변환
# ──────────────────────────────────────────────
do_pdf() {
    local midi_file="$1"
    shift || true

    if [ -z "$midi_file" ]; then
        echo "❌ 오류: MIDI 파일 경로를 입력해주세요."
        echo "   예: bash run_all.sh pdf ./output/song.mid"
        return 1
    fi

    python -m mt3.midi_to_pdf "$midi_file" "$@"
}

# ──────────────────────────────────────────────
# 메인 분기
# ──────────────────────────────────────────────
case "${1}" in
    train)
        shift
        do_train "$@"
        ;;
    infer)
        shift
        do_infer "$@"
        ;;
    pdf)
        shift
        do_pdf "$@"
        ;;
    install)
        do_install
        ;;
    -h|--help|help)
        usage
        ;;
    *)
        # 인자 없이 실행하거나 파일 경로만 넘긴 경우 → 추론 모드
        if [ -n "$1" ] && [ -f "$1" ]; then
            do_infer "$1"
        else
            usage
        fi
        ;;
esac
