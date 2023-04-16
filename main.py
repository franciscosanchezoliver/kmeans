import pandas as pd

# Example on how to use KMeans
# EXT. ABANDONED CITY - DAY
#
# Two zombies, ROTTEN and DECAYED, wander aimlessly until they lock eyes and fall in love.
# But when a group of human survivors threatens to tear them apart, ROTTEN and DECAYED must
# fight to stay together and prove that love can conquer all, even death.
#
# FADE TO BLACK.

# We want to create 3 groups dependen
# We like zombies movies and we would like to classify them into 2 groups, why? who knows...
# This will be our dataset

# | Movie                        | Zombies killed | Laughs per minute |
# |------------------------------|----------------|-------------------|
# | Shaun of the Dead            |             72 |                 4 |
# | Zombieland                   |            122 |                 7 |
# | Warm Bodies                  |             40 |                 3 |
# | The Return of the Living Dead|            186 |                 2 |
# | Army of Darkness             |             50 |                 6 |
# | Dead Alive                   |            321 |                 8 |
# | Night of the Creeps          |             28 |                 5 |
# | Dance of the Dead            |             16 |                 7 |
# | The Cabin in the Woods       |             18 |                 6 |
# | One Cut of the Dead          |             42 |                 9 |
# | Cockneys vs Zombies          |             63 |                 4 |
# | The Dead Don't Die           |             28 |                 3 |


data = {
    'Movie': ['Shaun of the Dead', 'Zombieland', 'Warm Bodies', 'The Return of the Living Dead',
              'Army of Darkness', 'Dead Alive', 'Night of the Creeps', 'Dance of the Dead',
              'The Cabin in the Woods', 'One Cut of the Dead', 'Cockneys vs Zombies', 'The Dead Don\'t Die'],
    'Year': [2004, 2009, 2013, 1985, 1992, 1992, 1986, 2008, 2012, 2017, 2012, 2019],
    'Zombies killed': [72, 122, 40, 186, 50, 321, 28, 16, 18, 42, 63, 28],
    'Laughs per minute': [4, 7, 3, 2, 6, 8, 5, 7, 6, 9, 4, 3]
}

df = pd.DataFrame(data)
print(df)
