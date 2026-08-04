import dotenv
import wandb

dotenv.load_dotenv()

run = wandb.init(project="uj-pr-individual-identification-project", name="initial-test-run")

run.log({"message": "Hello, World!"})

run.finish()