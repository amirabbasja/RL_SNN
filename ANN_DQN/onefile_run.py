# Runs the training script with desired parameters
import subprocess
import os
import time
import sys

startTime = time.time()
endTime = startTime + 4 * 60 * 60  # 4 hours
maxRunTime = 60 * 60 # 1 hour

trainingEpoch = 1
while time.time() < endTime:
    # Run parameters
    argsDict = {
        "name": "parallelDQN",
        "continue_run": True,
        "agents": 1,
        "hidden_layers": [64, 64],
        "learning_rate": 0.0001,
        "decay": 0.999,
        "batch": 1000,
        "gamma": 0.995,
        "extra_info": "",
        "max_run_time": maxRunTime, # In seconds
    }

    # For passing the args to the script
    scriptArgs = []
    for name, value in argsDict.items():
        if name == "continue_run":
            scriptArgs.extend([f"--continue_run"]) if value else None
            continue
        
        if name == "hidden_layers":
            scriptArgs.extend([f"--{name}"] + [str(l) for l in value])
        else:
            scriptArgs.extend([f"--{name}", str(value)])

    venvPath = "C:/Users/Spino.shop/Desktop/Trading/Apps/.MotherVenv/Scripts/python.exe"
    scriptPath = "./ANN_DQN/ANN_EXAMPLE_2_onefile.py"

    command = [venvPath, scriptPath] + scriptArgs

    try:
        # Set environment to force unbuffered output
        env = os.environ.copy()
        env['PYTHONUNBUFFERED'] = '1'
        
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,  # Merge stderr with stdout
            text=True,
            bufsize=0,
            universal_newlines=True,
            env=env
        )

        # Read output line by line in real-time
        for i, line in enumerate(iter(process.stdout.readline, '')):
            print(line, end='', flush=True)
            
        process.wait()
        
        if process.returncode != 0:
            print(f"Script failed with return code {process.returncode}")
    except FileNotFoundError:
        print(f"Virtual environment Python or script not found. Check paths: {venvPath}, {scriptPath}")
    except Exception as e:
        print(f"Error running script: {e}")
    
    trainingEpoch += 1

print(f"Reached the maximum run time. Trained {trainingEpoch} epochs")