cargo run --release -- --model all-MiniLM-L6-v2 --max-seq-length 256 --device cpu --embedding-dim 384 --batch-size 8 --num-runs 13
cargo run --release -- --model all-MiniLM-L6-v2 --max-seq-length 256 --device cpu --embedding-dim 384 --batch-size 32 --num-runs 13
cargo run --release -- --model all-MiniLM-L6-v2 --max-seq-length 256 --device cpu --embedding-dim 384 --batch-size 64 --num-runs 13
cargo run --release -- --model all-MiniLM-L6-v2 --max-seq-length 256 --device cpu --embedding-dim 384 --batch-size 128 --num-runs 13
