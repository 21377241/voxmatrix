import json
import select
import sys
import traceback


def serve(handler):
    while True:
        line = sys.stdin.readline()
        if not line:
            return
        anchor = line.find("->")
        if anchor < 0:
            print("Error: Invalid request format", flush=True)
            continue
        prefix = line[: anchor + 2]
        try:
            payload = json.loads(line[anchor + 2 :])
            result = handler(payload)
            print(f"{prefix}{json.dumps(result, ensure_ascii=False)}", flush=True)
            readable, _, _ = select.select([sys.stdin], [], [], 60)
            if readable:
                acknowledgment = sys.stdin.readline().strip()
                if acknowledgment != f"{prefix}close":
                    print(
                        f"Error: Unexpected acknowledgment: {acknowledgment}",
                        flush=True,
                    )
        except Exception as error:
            traceback.print_exc(file=sys.stderr)
            print(f"Error: {error}", flush=True)
