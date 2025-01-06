# Task-Oriented Dialogue System Evaluation Script

This directory is designed to evaluate a task-oriented dialogue system by generating synthetic conversations, extracting task completion metrics, and producing a labeled synthetic dataset. The evaluation can be run by running `eval.py`. The inputs and outputs to the script are shown below.

---

## **Inputs**

1. **Configuration File (`--config`)**  
   - A JSON configuration for the bot.
   - Example path: `./example/customer_service_config.json`.

2. **Model API (`--model_api`)**  
   - URL of the API endpoint for the dialogue model to be evaluated.  
   - Example: `http://myserver.com/eval/chat`.

3. **Model Parameters (`--model_params`)**  
   - Dictionary containing any additional parameters for the dialogue model (optional).  
   - Example: `{}`.

4. **Synthetic Data Parameters**  
   These parameters control the generation of synthetic conversations:  
   - `--num_convos`: Number of synthetic conversations to simulate.  
   - `--num_goals`: Number of goals/tasks to simulate.  
   - `--max_turns`: Maximum number of turns per conversation.  

5. **Documents Directory (`--documents_dir`)**  
   - Path to the directory containing the relevant documents for the bot.
   - Example: `example/customer_service`.

---

## **Outputs**

1. **Simulated Synthetic Dataset (`simulate_data.json`)**  
   - JSON file containing simulated conversations generated based on the user's objective to evaluate the task success rate.
  
2. **Labeled Synthetic Dataset (`labeled_data.json`)**  
   - JSON file containing labeled conversations generated based on the taskgraph to evaluate the NLU performance.

3. **Goal Completion Metrics (`goal_completion.json`)**  
   - JSON file summarizing task completion statistics based on the bot's ability to achieve specified goals.
