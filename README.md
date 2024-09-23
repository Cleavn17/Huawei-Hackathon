# Running the Naive Solution

## VENV Stuff

```
… Install requirements.txt in a VENV / CONDA / whatever tickles your pickle
```

## Preparing Solutions

Make sure an empty 'output' folder exists.

```
NAIVE_LOG_LEVEL=INFO <test_seeds xargs -P 10 -I {} python3 mysolution.py --seed {}
<test_seeds xargs -P 10 -I {} python3 evaluation_example.py output/{}.json --silent --seed {}
```

Zip the seeds (?). IDK how to do this not going to lie.

```
zip seeds.zip ./output/*.json
rm code.zip ; zip -r code.zip mysolution.py timemachine.py test_seeds data naive.py utils.py seeds.py requirements.txt evaluation.py evaluation_example.py matrix.py README.md
```

## Hyperparameter Tuning

This is really an end-game strategy when you can't think of other ways to speed up your code:

```
NAIVE_LOG_LEVEL=ERROR TIME_MACHINE_LOG_LEVEL=ERROR <test_seeds xargs -P 10 -I {}  python3 matrix.py --session durk --silent --seed {}
```

Hyper parameter tuning should increase your result by about 2% after about 3 hours. The efficacy of this approach is directly proportional to how fast the solution runs and how fast evaluation runs. The method used is grid search.

I reccomend using a codespace in Microsoft GitHub for doing hyper parameter search and copying the files over intermittently.

```
while : ; do ssh codespace 'cd /workspaces/Huawei-Hackathon/kurd/solutions/ ; { cat ../reports/*.json | jq .score ; } | paste -sd + | bc ; rm -f seeds.zip ; >/dev/null zip -r9 seeds.zip *.json' ; rsync codespace:/workspaces/Huawei-Hackathon/kurd/solutions/seeds.zip /tmp/ ; sleep 30 ; done
```

## Using the Time Machine

```
python3 mysolution.py --seed 123 --mutate output/123.json
```

The time machine doesn't really work as well as I thought it would (it never worked at all as we don't have proper profit accounting).

## Plotting

```
python3 plot.py
```
