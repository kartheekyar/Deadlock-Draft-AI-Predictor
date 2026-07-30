# Deadlock-Draft-AI-Predictor
Discord Bot that find win probability based on draft picks.


# Deadlock AI Draft Assistant

A Discord bot that uses an AI model to look at a 6v6 character draft in the game *Deadlock* and predict which team is more likely to win.

---

## 🤔 What is this?
When playing Deadlock, team composition matters. This bot allows players to type their team and the enemy team into Discord. 

The AI looks at the matchups, checks if the teams have a good balance of roles (like Tanks and Supports), calculates combo synergies (like Seven + Dynamo), and instantly outputs a win percentage.

---

## 🚀 How to Run This Project

### Step 1: Install the Code Libraries
Open your computer's terminal or command prompt and run this command to download the tools the code needs:
```bash
pip install discord.py pandas scikit-learn requests pip_system_certs
```

### Step 2: How to Find Your Secret Discord Bot Key
To let the code control your bot, you need to grab its private token password:
1. Go to the [Discord Developer Portal](https://discord.com) and sign in.
2. Click the green **New Application** button in the top right corner and name it.
3. On the left side menu, click the **Bot** tab (has a little robot head icon).
4. Scroll down to the middle of that page and click the blue **Reset Token** button. 
5. Click **Copy** to save that long string of random text. This string is your `BOT_TOKEN` key!
6. *Crucial:* Scroll a little further down on that same page, find **Message Content Intent**, flip the switch to **On**, and click save.

### Step 3: Add the Key to Your Code
Open `deadlock_bot.py` on your computer, scroll to the very bottom, and paste your key directly inside the quotation marks:
```python
BOT_TOKEN = "PASTE_YOUR_DISCORD_KEY_HERE"
```

### Step 4: Launch the Bot
Run the `deadlock_bot.py` script inside PyCharm or your terminal. Go to your server and type:

`!predict Seven, Dynamo, Haze, Abrams, Wraith, Ivy vs Bebop, Infernus, Warden, Paradox, Lash, Vindicta`
