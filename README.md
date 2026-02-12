# crabingest

Run without cloning:

`python3 <(curl -fsSL https://raw.githubusercontent.com/SauersML/crabingest/main/grab.py) SauersML/Reagle`

Run without cloning (with include pattern):

`python3 <(curl -fsSL https://raw.githubusercontent.com/SauersML/crabingest/main/grab.py) SauersML/Reagle "src/**"`

Run without cloning (multiple include patterns: comma-separated):

`python3 <(curl -fsSL https://raw.githubusercontent.com/SauersML/crabingest/main/grab.py) SauersML/Reagle "src/**,crates/**,examples/**"`

Use multiple excludes by repeating `--exclude`:

`python3 <(curl -fsSL https://raw.githubusercontent.com/SauersML/crabingest/main/grab.py) SauersML/Reagle "src/**,crates/**" --exclude "tests/**" --exclude "**/*_test.rs"`
