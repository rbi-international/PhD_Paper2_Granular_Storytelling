"""
fetch_shard.py

Layman note: a stubborn file downloader. It asks the server for the next chunk, writes
it straight to disk, and keeps going until the file is whole. If the connection drops
it picks up from the exact byte it reached, so nothing already downloaded is lost.

Why this exists: huggingface_hub's snapshot_download hung after a network change,
writing zero bytes for minutes at a time, while a plain HTTP range request from the
same offset returned data immediately (HTTP 206). The library was stuck on a dead
connection, not blocked by the network. This bypasses it for the one shard that is
incomplete, then hands control back to the normal cache layout.

Once the blob is whole and renamed, huggingface_hub sees it as cached and will simply
create the snapshot symlink on the next call, so the model loads normally afterwards.

Usage:
    python src/6_ablations/fetch_shard.py

No em dashes anywhere (project style rule).
"""
import os
import time
import urllib.request

REPO = "microsoft/Phi-3-mini-4k-instruct"
SHARD = "model-00001-of-00002.safetensors"
URL = f"https://huggingface.co/{REPO}/resolve/main/{SHARD}"
BLOBS = os.path.join(
    os.path.expanduser("~"), ".cache", "huggingface", "hub",
    f"models--{REPO.replace('/', '--')}", "blobs",
)

CHUNK = 16 * 1024 * 1024   # 16 MB per request, small enough to retry cheaply
MAX_RETRIES = 200          # the connection is flaky, not broken, so retry patiently
TIMEOUT = 60


def expected_size():
    request = urllib.request.Request(URL, method="HEAD")
    with urllib.request.urlopen(request, timeout=TIMEOUT) as response:
        return int(response.headers["Content-Length"])


def find_partial():
    """The .incomplete blob for this shard, or the finished blob if already done."""
    for name in os.listdir(BLOBS):
        path = os.path.join(BLOBS, name)
        if name.endswith(".incomplete") and os.path.getsize(path) > 1e9:
            return path
    return None


def main():
    total = expected_size()
    partial = find_partial()
    if partial is None:
        print("No large .incomplete blob found. Shard may already be complete.")
        return

    print(f"Target : {SHARD}")
    print(f"Size   : {total:,} bytes")
    print(f"Have   : {os.path.getsize(partial):,} bytes")
    print(f"Need   : {total - os.path.getsize(partial):,} bytes\n")

    consecutive_failures = 0
    started = time.time()
    start_bytes = os.path.getsize(partial)

    while True:
        have = os.path.getsize(partial)
        if have >= total:
            break
        end = min(have + CHUNK - 1, total - 1)
        request = urllib.request.Request(URL, headers={"Range": f"bytes={have}-{end}"})
        try:
            with urllib.request.urlopen(request, timeout=TIMEOUT) as response:
                data = response.read()
            if not data:
                raise IOError("empty chunk")
            # Append and fsync, so a crash never costs more than one chunk.
            with open(partial, "ab") as handle:
                handle.write(data)
                handle.flush()
                os.fsync(handle.fileno())
            consecutive_failures = 0
            done = os.path.getsize(partial)
            elapsed = max(time.time() - started, 1)
            rate = (done - start_bytes) / elapsed / 1048576 * 60
            remaining = (total - done) / 1048576 / max(rate, 0.1)
            print(f"  {done/1048576:7.0f} / {total/1048576:.0f} MB "
                  f"({done/total*100:5.1f}%)  {rate:6.1f} MB/min  "
                  f"~{remaining:5.1f} min left", flush=True)
        except Exception as error:
            consecutive_failures += 1
            if consecutive_failures > MAX_RETRIES:
                print(f"\nGiving up after {MAX_RETRIES} consecutive failures: {error}")
                return
            wait = min(2 ** min(consecutive_failures, 5), 30)
            print(f"  retry {consecutive_failures}: {type(error).__name__}, "
                  f"waiting {wait}s", flush=True)
            time.sleep(wait)

    final = partial[:-len(".incomplete")]
    os.replace(partial, final)
    print(f"\nCOMPLETE. {os.path.getsize(final):,} bytes")
    print(f"Renamed to blob: {os.path.basename(final)}")
    print("huggingface_hub will now see this as cached and only needs to symlink it.")


if __name__ == "__main__":
    main()
