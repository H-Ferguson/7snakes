# 7snakes? 

Started as a November 2022 Hackathon Project

## Building 
This code has a dependancy on a modified fork of a Battlesnake PettingZoo Environment I found https://github.com/DaBultz/pz-battlesnake/, 
the fork to use is here https://github.com/H-Ferguson/pz-battlesnake

Install the forked `pz-battlesnake` library locally by running the `setup.py` script that lives in that repo: 
```
python3 setup.py install --user
``` 
make sure to install other dependancies using pip (now back in this repo):
```
pip install -r requirements.txt
```

You can then run the working trainer that lives in `diy_trainer.py`

## TODO
- Probably write tests 
- There are still some edge case bugs that are preventing the model from training for long, those need to be sniffed out
- Make sure the model is actually learning something? 
- I think the learning algorithm itself may need tweaking, Need to read up on the math behind [DQN](https://arxiv.org/pdf/1509.06461.pdf) more to be able to identify what needs to be done
- Write out the model to a pickle file and add infra to load that model back in for actual gameplay
- Write server endpoints that query the policies generated by the trainer (unless I finished them and didn't update this readme)

# Battlesnake Python Starter Project

An official Battlesnake template written in Python. Get started at [play.battlesnake.com](https://play.battlesnake.com).

![Battlesnake Logo](https://media.battlesnake.com/social/StarterSnakeGitHubRepos_Python.png)

This project is a great starting point for anyone wanting to program their first Battlesnake in Python. It can be run locally or easily deployed to a cloud provider of your choosing. See the [Battlesnake API Docs](https://docs.battlesnake.com/api) for more detail. 

[![Run on Replit](https://repl.it/badge/github/BattlesnakeOfficial/starter-snake-python)](https://replit.com/@Battlesnake/starter-snake-python)

## Technologies Used

This project uses [Python 3](https://www.python.org/) and [Flask](https://flask.palletsprojects.com/). It also comes with an optional [Dockerfile](https://docs.docker.com/engine/reference/builder/) to help with deployment.

## Run Your Battlesnake

Install dependencies using pip

```sh
pip install -r requirements.txt
```

Start your Battlesnake

```sh
python main.py
```

You should see the following output once it is running

```sh
Running your Battlesnake at http://0.0.0.0:8000
 * Serving Flask app 'My Battlesnake'
 * Debug mode: off
```

Open [localhost:8000](http://localhost:8000) in your browser and you should see

```json
{"apiversion":"1","author":"","color":"#888888","head":"default","tail":"default"}
```

## Play a Game Locally

Install the [Battlesnake CLI](https://github.com/BattlesnakeOfficial/rules/tree/main/cli)
* You can [download compiled binaries here](https://github.com/BattlesnakeOfficial/rules/releases)
* or [install as a go package](https://github.com/BattlesnakeOfficial/rules/tree/main/cli#installation) (requires Go 1.18 or higher)

Command to run a local game

```sh
battlesnake play -W 11 -H 11 --name 'Python Starter Project' --url http://localhost:8000 -g solo --browser
```

## Next Steps

Continue with the [Battlesnake Quickstart Guide](https://docs.battlesnake.com/quickstart) to customize and improve your Battlesnake's behavior.

**Note:** To play games on [play.battlesnake.com](https://play.battlesnake.com) you'll need to deploy your Battlesnake to a live web server OR use a port forwarding tool like [ngrok](https://ngrok.com/) to access your server locally.
