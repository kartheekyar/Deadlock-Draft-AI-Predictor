import ssl

# FORCE GLOBAL MAC FIX: This globally turns off certificate checks before anything else loads
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

import discord
from discord.ext import commands
import numpy as np

# Import the functions directly from your first AI file
from deadlock_draft_ai import fetch_deadlock_heroes, generate_meta_training_data, train_ai_model, predict_draft_win_rate

# 1. SETUP THE BOT INTERFACE WITH INTENTS
intents = discord.Intents.default()
intents.message_content = True
bot = commands.Bot(command_prefix="!", intents=intents)

# Global variables to store our trained model across chat requests
AI_MODEL = None
HERO_MAP = None


@bot.event
async def on_ready():
    """Triggers automatically when the bot connects to Discord."""
    global AI_MODEL, HERO_MAP
    print(f"[+] Bot logged in successfully as {bot.user}")
    print("[*] Actively caching Deadlock meta nodes and training engine...")

    # Pre-train the model immediately on startup so chat commands respond instantly
    HERO_MAP = fetch_deadlock_heroes()
    dataset = generate_meta_training_data(HERO_MAP, num_matches=4000)
    AI_MODEL = train_ai_model(dataset)

    print("[+] AI Brain loaded! The bot is ready to process match drafts.")


# 2. CREATE THE USER COMMAND
@bot.command(name="predict")
async def predict(ctx, *, draft_input: str = None):
    """
    User command format: !predict Ally1, Ally2... vs Enemy1, Enemy2...
    Parses live player text, runs the AI model, and prints the result back.
    """
    if not draft_input:
        await ctx.send(
            "❌ **Error:** Please provide team compositions!\n**Format:** `!predict Hero1, Hero2, Hero3, Hero4, Hero5, Hero6 vs Enemy1, Enemy2, Enemy3, Enemy4, Enemy5, Enemy6`")
        return

    # Check for the separator 'vs'
    if " vs " not in draft_input.lower():
        await ctx.send(
            "❌ **Error:** You must split the teams using `vs`!\n**Format:** `!predict [Your Team] vs [Enemy Team]`")
        return

    try:
        # Split string input into two distinct team blocks
        ally_part, enemy_part = draft_input.lower().split(" vs ")

        # Clean lists by stripping whitespace and removing empty items
        ally_team = [hero.strip() for hero in ally_part.split(",") if hero.strip()]
        enemy_team = [hero.strip() for hero in enemy_part.split(",") if hero.strip()]

        # Validation checks to match game parameters
        if len(ally_team) != 6 or len(enemy_team) != 6:
            await ctx.send(
                f"❌ **Draft Error:** Deadlock requires exactly **6** players per team.\nDetected: Blue ({len(ally_team)}/6) vs Orange ({len(enemy_team)}/6)")
            return

        # Let users know the AI is processing the combination
        await ctx.send("🔮 *Analyzing team drafting vectors and calculating synergies...*")

        # Execute our machine learning prediction loop
        raw_probabilities = predict_draft_win_rate(AI_MODEL, HERO_MAP, ally_team, enemy_team)

        # FIX: Explicitly flatten out the numpy matrix layout array down to a direct value
        win_rate_value = float(np.squeeze(raw_probabilities))

        # Format a clean, visually structured message response block
        embed = discord.Embed(title="Deadlock Match Synergy Report", color=0x3498db)
        embed.add_field(name="🟦 Blue Squad (Ally)", value=", ".join([h.title() for h in ally_team]), inline=False)
        embed.add_field(name="🟧 Orange Squad (Enemy)", value=", ".join([h.title() for h in enemy_team]), inline=False)
        embed.add_field(name="📊 AI Calculated Win Probability", value=f"**{win_rate_value:.2f}%**", inline=False)

        await ctx.send(embed=embed)

    except Exception as e:
        await ctx.send(
            f"⚠️ **Processing Error:** An issue occurred evaluating your text profile. Ensure hero spelling matches game names.")
        print(f"Error parsing bot text: {e}")


# 3. RUN THE BOT ENGINE
BOT_TOKEN = "YOUR_PRIVATE_TOKEN_HERE"
if BOT_TOKEN == "YOUR_DISCORD_BOT_TOKEN_HERE":  #Don't put token here
    print("[-] Startup stopped: You must generate and enter a Discord Bot Token inside deadlock_bot.py.")
else:
    bot.run(BOT_TOKEN)
