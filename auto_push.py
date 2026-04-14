import git
import schedule
import time
from datetime import datetime
import os
import sys

# Install required libraries if not present
try:
    import schedule
    import git
except ImportError:
    print("Installing required libraries...")
    os.system('pip install schedule GitPython')
    import schedule
    import git

# Set up repository
repo_path = os.path.dirname(os.path.abspath(__file__))
repo = git.Repo(repo_path)

# Get GitHub token from command line argument or prompt
if len(sys.argv) > 1:
    token = sys.argv[1]
else:
    token = input('Enter your GitHub Personal Access Token: ')

# Update remote URL with token
origin = repo.remote('origin')
origin.set_url(f'https://sunil448832:{token}@github.com/sunil448832/object-tracking.git')

def commit_and_push():
    try:
        # Check for changes
        if repo.is_dirty() or repo.untracked_files:
            repo.git.add(A=True)
            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            repo.index.commit(f'Auto commit at {timestamp}')
            origin.push()
            print(f'Committed and pushed changes at {timestamp}')
        else:
            print('No changes to commit and push')
    except Exception as e:
        print(f'Error during commit/push: {e}')

# Schedule the job every 15 minutes
schedule.every(15).minutes.do(commit_and_push)

print('Auto-push script started. It will commit and push changes every 15 minutes.')
print('Press Ctrl+C to stop the script.')

# Run the scheduler
while True:
    schedule.run_pending()
    time.sleep(1)