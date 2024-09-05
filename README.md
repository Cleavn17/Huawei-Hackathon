# Running the Naive Solution

## VENV Stuff

```
… Install requirements.txt in a VENV / CONDA / whatever tickles your pickle
```

## Preparing Solutions

Make sure an empty 'output' folder exists.

```
mkdir -p output
NAIVE_LOG_LEVEL=INFO <test_seeds xargs -P 10 -I {} python3 mysolution.py --seed {}
<test_seeds xargs -P 10 -I {} python3 evaluation_example.py output/{}.json --silent --seed {}
```

Zip the seeds (?). IDK how to do this not going to lie.

```
zip seeds.zip ./output/*.json
zip -r code.zip ./*
```
