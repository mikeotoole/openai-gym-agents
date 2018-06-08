Agent to solve Cart Pole environment

### Setup

```bash
brew cask install miniconda
export PATH=/usr/local/miniconda3/bin:$PATH # Likely want to add this to your .bash_profile
conda install --quiet --yes tensorflow
pip install keras==2.0.8
pip install gym==0.9.4
pip install h5py
```

### Run

```bash
python cart_pole/number_one.py
```

You will see the score output for each episode. Largest score is 200 (so you
will see a max of 199).
