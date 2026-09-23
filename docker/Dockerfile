# MiMo-V2.6-Pro-RL on 8 x DGX Spark (GB10, sm_121a) with SGLang.
#
# Base: lmsysorg/sglang dev-cu13 nightly 2026-09-23 (sglang 06008c17), which
# carries sgl-project/sglang#40448 (MiMo-V2.6 day-0: packed MXFP4 experts,
# DFlash). Older images expand the experts to FP8 and exhaust a Spark.
ARG BASE_IMAGE=lmsysorg/sglang@sha256:5921361fcf358cdde4df1968c941c14157f418613b099ad7f3e5aeed6427ae15
FROM ${BASE_IMAGE}
ARG BASE_IMAGE
LABEL mimo26.base="${BASE_IMAGE}" mimo26.sglang_base_commit="06008c170dae333e22caceb51f36224071f9b5b3"

# Vision/video preprocessing (MiMo processor shells out to ffprobe; audio uses torchcodec).
RUN apt-get update \
 && apt-get install -y --no-install-recommends ffmpeg \
 && rm -rf /var/lib/apt/lists/* \
 && pip install --no-cache-dir torchcodec

# Frozen b12x snapshot qualified on this fabric for DeepSeek SG17 (Apache-2.0).
COPY third_party/b12x/ /opt/b12x/

# Unmerged upstream PRs, applied in order on top of the base commit.
COPY patches/sglang/ /opt/mimo26/patches/sglang/
# rocenante: SGLang hook for the b12x RoCE one-shot all-reduce (ported from the
# DeepSeek V4.1 SG17 build; enabled at runtime with SGLANG8_ROCE_ALLREDUCE=1).
# mimo26-fast-expert-load: skip experts owned by other EP ranks before reading them.
# mimo26-marlin-skip-zero-fill: no zero-fill of Marlin MoE scratch when every expert is local.
ARG SGLANG_PRS="40722 40888 38142 40633 40665 29858 rocenante mimo26-fast-expert-load mimo26-marlin-skip-zero-fill"
RUN cd /sgl-workspace/sglang \
 && test "$(git rev-parse HEAD)" = 06008c170dae333e22caceb51f36224071f9b5b3 \
 && for pr in ${SGLANG_PRS}; do echo "applying $pr"; git apply /opt/mimo26/patches/sglang/$pr.diff || exit 1; done \
 && printf '%s\n' ${SGLANG_PRS} > /opt/mimo26/applied-prs.txt \
 && PYTHONPATH=/opt/b12x python3 -c "import sglang.srt.models.mimo_v2, sglang.srt.speculative.dflash_worker_v2, sglang.srt.distributed.device_communicators.pynccl; print('import ok')"

COPY runtime/ /opt/mimo26/runtime/

ENV TORCH_CUDA_ARCH_LIST=12.1a \
    FLASHINFER_CUDA_ARCH_LIST=12.1a \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:False \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/opt/b12x \
    B12X_ROCE_CACHE_DIR=/root/.cache/b12x-roce \
    SGLANG_RUST_BUILD_MODE=never

ENTRYPOINT []
