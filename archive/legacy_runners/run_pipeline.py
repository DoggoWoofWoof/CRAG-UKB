import subprocess
import re
import sys
import os

def run_filtered_modal():
    dataset = "SQuAD, MuSiQue, 2Wiki"
    log_filename = "pipeline.log"
    
    # Use sys.executable to run modal as a module.
    cmd = [sys.executable, "-m", "modal", "run", "run_modal.py"]
    
    # Forward all CLI arguments
    if len(sys.argv) > 1:
        cmd.extend(sys.argv[1:])
        
    dataset = "Specified from args" if "--dataset" in sys.argv else "SQuAD, MuSiQue, 2Wiki, MetaQA"

    # Spam patterns to ignore
    ignore_patterns = [
        r"Creating objects", r"Creating mount", r"Uploaded", r"Finalizing index",
        r"Creating function", r"Created objects", r"Initializing...", 
        r"Running app", r"Worker assigned", r"Loading images", 
        r"Running \(\d+/\d+ containers active\)",
        r"Created mount", r"Created function",
        r"Mounting .+",
        r"Connecting from Modal", r"keyboard interrupt",
        r"0/1 \[00:00", r"1/1 \[00:00", r"0/2 \[00:00", r"2/2 \[00:00",
        r"Batches:.*\b0/1\b", r"Batches:.*\b1/1\b", r"Batches:   0%\|", r"Batches: 100%\|"
    ]

    # Regex for lines that start with a spinner
    spinner_start = r"^[|/\\-]\s"

    print(f"🚀 Launching Modal app for {dataset}...")
    print(f"📋 Logging clean output to {log_filename}")
    print(f"🔧 Command: {' '.join(cmd)}")

    url_printed = False

    # Prepare environment with UTF-8 enforcement for the child process
    env = os.environ.copy()
    env["PYTHONUTF8"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"
    # Some CLI tools also look at these
    env["LC_ALL"] = "en_US.UTF-8"
    env["LANG"] = "en_US.UTF-8"

    with open(log_filename, "w", encoding="utf-8") as f:
        process = subprocess.Popen(
            cmd, 
            stdout=subprocess.PIPE, 
            stderr=subprocess.STDOUT, 
            text=True, 
            encoding='utf-8', 
            errors='replace', # Be resilient to decoding errors
            bufsize=1,
            env=env
        )
        
        if process.stdout is None:
            print(f"❌ Error: Could not capture output from the process.")
            return

        while True:
            raw_line = process.stdout.readline()
            if not raw_line:
                if process.poll() is not None:
                    break
                continue

            # 1. Strip ALL ANSI escape sequences (cursor movement, colors, etc.)
            clean = re.sub(r'\x1b\[[0-9;]*[A-Za-z]', '', raw_line)
            
            # 2. Split by \r — tqdm uses carriage returns to overwrite lines.
            #    Take the LAST non-empty segment (what would be visible on a real terminal).
            segments = clean.split('\r')
            line = ''
            for seg in reversed(segments):
                seg = seg.strip()
                if seg:
                    line = seg
                    break
            
            # 3. Skip completely empty lines
            if not line:
                continue

            # 4. Check for URL
            if "View app at" in line or "modal.com" in line:
                if not url_printed:
                    f.write(line + '\n')
                    f.flush()
                    print(f"🔗 {line}")
                    url_printed = True
                continue 

            # 5. Check against ignore patterns
            if any(re.search(p, line) for p in ignore_patterns):
                continue

            # 6. Filter out spinner lines
            if re.match(spinner_start, line):
                continue
                
            # 7. Write clean line to log file
            f.write(line + '\n')
            f.flush()
            
            # 8. Mirror to terminal safely
            try:
                print(line)
            except UnicodeEncodeError:
                print(line.encode(sys.stdout.encoding, errors='replace').decode(sys.stdout.encoding))

    process.wait()
    print("\n✅ Pipeline completed.")

if __name__ == "__main__":
    run_filtered_modal()
