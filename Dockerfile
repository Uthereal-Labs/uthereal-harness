# syntax=docker/dockerfile:1.7

FROM rust:1.96.1-bookworm AS builder

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential ca-certificates clang cmake pkg-config protobuf-compiler \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build
COPY . .

ENV CARGO_INCREMENTAL=0 \
    CARGO_BUILD_JOBS=2 \
    CARGO_PROFILE_RELEASE_STRIP=symbols \
    CARGO_REGISTRIES_CRATES_IO_PROTOCOL=sparse

RUN --mount=type=cache,id=goose-cargo-registry,target=/usr/local/cargo/registry \
    --mount=type=cache,id=goose-cargo-git,target=/usr/local/cargo/git \
    --mount=type=cache,id=goose-target,target=/build/target \
    cargo build --locked --release -p goose-cli --bin goose \
        --no-default-features --features rustls-tls,otel,disable-update \
    && install -Dm755 target/release/goose /out/goose

FROM debian:bookworm-slim@sha256:b1a741487078b369e78119849663d7f1a5341ef2768798f7b7406c4240f86aef AS runtime

RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates curl git ripgrep \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --create-home --uid 10001 --shell /bin/bash goose \
    && install -d -o goose -g goose /var/lib/goose /workspace

COPY --from=builder /out/goose /usr/local/bin/goose

ENV GOOSE_PATH_ROOT=/var/lib/goose \
    GOOSE_DISABLE_KEYRING=1

USER goose
WORKDIR /workspace
VOLUME ["/var/lib/goose", "/workspace"]
EXPOSE 3284

HEALTHCHECK --interval=15s --timeout=3s --start-period=10s --retries=3 \
    CMD ["curl", "--fail", "--silent", "--show-error", "http://127.0.0.1:3284/health"]

ENTRYPOINT ["/usr/local/bin/goose"]
CMD ["serve", "--host", "0.0.0.0", "--port", "3284", "--platform", "cli", "--with-builtin", "developer,summon"]

LABEL org.opencontainers.image.title="uthereal goose harness" \
      org.opencontainers.image.description="Native ACP server for the Uthereal goose harness" \
      org.opencontainers.image.source="https://github.com/Uthereal-Labs/uthereal-harness"
