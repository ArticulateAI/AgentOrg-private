# Customer Service Bot

*Enhance your bots with Retrieval Augemnted Generation (RAG)*

## Intro

In this tutorial, we'll walk through building a basic customer service bot using **Agent First Organization**'s framework. This bot will be able to handle common customer inquiries, such as answering FAQs, identifying customer preferences, and request customer's contact information. The tutorial aims to build on top of a [simple conversational AI](./roleplay-chatbot.md) to include RAG in order to equip you with the foundational skills needed to create and deploy a functional customer service bot.

By the end of this tutorial, you'll know how to set up the AI framework, prepare documents for RAG, build a basic conversation flow, and power a bot with it. Whether you're building your first bot or refining your skills, this tutorial will guide you in creating a responsive and helpful customer service chatbot. 

## Supplying Information

Before any technical work is done, we must first identify and find information to supply to the bot. After all, you can't serve as a Customer Service Agent without anything to support! To add resources to the bot, we need to decide what the bot should known and look for links to sources in which the bot can learn from. E.g, a bunch of wikipedia content. In this example, we will be using a companies' website to supply to the bot. Keep it saved somewhere, we will be needing it for our next step!

## Setting up the Config File

In its core, bot is powered through a *TaskGraph* which is the structure that links various tasks together to fulfill the overall role of the bot. Each "node" represents a task which has an *Worker* that is selected to complete task. Each node engages the user for their response, and  with the user response, the TaskGraph will decide which next node to travel to.

Like actual conversations, *TaskGraph* can be complicated; that is why we help you convert a simple and intuitive *Config* JSON file into a powerful and advanced *TaskGraph* through our generator. Instead of needing to design an entire graph, all you need to do is to describe the bot and provide some extra information and it will build the graph for you! 

#### Workers
Building on top of a primitive conversational bot, for the customer service bot, we need to let the bot be able to read from the documents we are supplying when composing a response. We can do that is through the [RAGWorker](../Workers/RAGWorker.mdx) which retrieves the relevant information from our sources and then passes the information to [MessageWorker](../Workers/MessageWorker.mdx) which composes the response. 

As a refresher, here is the structure for a [Config](../Config.md) JSON file:

* `role (Required)`: The general "role" of the chatbot you want to create. For instance, "customer service assistant", "data analyst", "shopping assistant", etc.
* `user_objective (Required)`: The user's goal that the chatbot wants to achieve. Related to the user experience. Description in third person. For instance, "The customer service assistant helps users with customer service inquiries. It can provide information about products, services, and policies, as well as help users resolve issues and complete transactions."
* `builder_objective (Optional)`: The additional target you want the chatbot to achieve beyond the user's goal. Can contain hidden objectives or subtle objectives which is hidden from the user. Describe in third person. For instance, "The customer service assistant helps to request customer's contact information."
* `domain (Optional)`: The domain that you want to create the chatbot for. For instance, "robotics and automation", "Ecommerce", "Healthcare", etc.
* `intro (Optional)`: The introduction of the above domain that you want to create the chatbot for. It should contain the information about the domain, the products, services, and policies, etc.
* `task_docs (Optional, List[Dict])`: The documents resources for the taskgraph generation to create the chatbot. Each item in the list should contain the following fields:
    * `source (Required)`: The source url that you want the chatbot to refer to
    * `desc (Optional)` : Short description of the source and how it is used
    * `num (Optional)`: The number of websites that you want the chatbot to refer to for the source, defaults to one (only the url page)
* `rag_docs (Optional, List[Dict])`: If you want to use RAGWorker, then here indicates the documents for the RAG component of chatbot when running chatbot. Each item in the list should contain the following fields:
    * `source (Required)`: The source url that you want the chatbot to refer to
    * `desc (Optional)` : Short description of the source and how it is used
    * `num (Optional)`: The number of websites that you want the chatbot to refer to for the source, defaults to one (only the url page)
* `tasks (Optional, List(Dict))`: The pre-defined list of tasks that the chatbot need to handle. If empty, the system will generate the tasks and the steps to complete the tasks based on the role, objective, domain, intro and docs fields. The more information you provide in the fields, the more accurate the tasks and steps will be generated. If you provide the tasks, it should contain the following fields:
    * `task_name (Required, Str)`: The task that the chatbot need to handle
    * `steps (Required, List(Str))`: The steps to complete the task
* `workers (Required, List(WorkerClassName))`: The [Workers](Workers/Workers.md) pre-defined under `agentorg/workers` folder in the codebase that you want to use for the chatbot.

Now, lets see it with the Customer Service Bot example. Here, we have a sample Config file of a Customer Service Bot for a robotics company - [RichTech](https://www.richtechrobotics.com/).

```json
{
    "role": "customer service assistant",
    "user_objective": "The customer service assistant helps users with customer service inquiries. It can provide information about products, services, and policies, as well as help users resolve issues and complete transactions.",
    "builder_objective": "The customer service assistant helps to request customer's contact information.",
    "domain": "robotics and automation",
    "intro": "Richtech Robotics's headquarter is in Las Vegas; the other office is in Austin. Richtech Robotics provide worker robots (ADAM, ARM, ACE), delivery robots (Matradee, Matradee X, Matradee L, Richie), cleaning robots (DUST-E SX, DUST-E MX) and multipurpose robots (skylark). Their products are intended for business purposes, but not for home purpose; the ADAM robot is available for purchase and rental for multiple purposes. This robot bartender makes tea, coffee and cocktails. Richtech Robotics also operate the world's first robot milk tea shop, ClouTea, in Las Vegas (www.cloutea.com), where all milk tea beverages are prepared by the ADAM robot. The delivery time will be one month for the delivery robot, 2 weeks for standard ADAM, and two months for commercial cleaning robot. ",
    "task_docs": [{
        "source": "https://www.richtechrobotics.com/",
        "num": 20
    }],
    "rag_docs": [{
        "source": "https://www.richtechrobotics.com/",
        "num": 20
    }],
    "tasks": [],
    "workers": [
        "RAGWorker",
        "RagMsgWorker",
        "MessageWorker",
        "SearchWorker",
        "DefaultWorker"
    ]
}
```
With our Config in place, the vast majority of work is surprisingly already done! The rest is simply bringing the bot to life.


## Generating a TaskGraph

Now that we have a Config file, generating the graph is the easy part. All you need to do is run 

`python create.py --config ./examples/customer_service_config.json --output-dir ./examples/customer_service`

It will first enter into a *task planning* interactive panel where you can see the generated tasks this bot will handle and the following steps to complete the specific tasks. You can also modify the tasks and steps as needed. Once you are satisfied with result, you could press `s` to save the *task planning* file then it will further generate the final *TaskGraph* file. 

TaskGraph provides the graph that the bot will traverse through during the conversation. It provides a guideline for the conversation to make it more controllable and reliable. The details can be viewed at [here](../Taskgraph/Generation.md).

It will also prepare the documents for the RAG component of the bot. It will first crawl the websites content specified by the `rag_docs` in the config file and saved the documents into `documents.pkl` file. Later on, during the conversation, the bot will execute RAG Worker to retrieve relevent information of user's query from the documents to compose responses to the user.

## Running the Bot

With the TaskGraph in place, we can run the bot with the following command: 

`python run.py --input-dir ./examples/customer_service`

It will initialize the service (e.g. NLU apis) you need to start the bot and you can start interacting with it!

---

## Sample Conversation
> **ASSISTANT**: Hello! Welcome to our Customer Service Assistant. I'm here to provide you with information about our products, services, and policies. How can I assist you today?
> 
> **USER**: What products do you offer?
> 
> **ASSISTANT**:\
> We offer a range of innovative robots designed for business purposes. Our lineup includes:
> 
> 1. **Worker Robots:** 
>    - ADAM: A versatile robot capable of tasks like food and beverage preparation.
>    - ARM and ACE: Designed for various operational needs.
> 
> 2. **Delivery Robots:** 
>    - Matradee, Matradee X, Matradee L, and Richie: These robots are perfect for automating delivery tasks.
> 
> 3. **Cleaning Robots:** 
>    - DUST-E SX and DUST-E MX: Ideal for maintaining cleanliness in commercial spaces.
> 
> 4. **Multipurpose Robot:**
>    - Skylark: This robot can handle multiple tasks including easy ordering, vending, and cleaning operations.
> 
> If you have any specific questions about a product or need further assistance, feel free to ask!
> 
> **USER**: I am interested in Cleaning Robots, can you tell me mor eabout the differences between DUST-E SX and DUST-E MX?
> 
> **ASSISTANT**:\
> Certainly! Both the DUST-E SX and DUST-E MX are designed for professional cleaning, but they cater to different needs:
> 
> 1. **DUST-E SX:**
>    - Designed for smaller environments.
>    - It's a compact solution for medium to small-sized buildings.
>    - Offers the same high-quality cleaning as the MX but in a smaller size, making it ideal for tighter or more confined spaces.
> 
> 2. **DUST-E MX:**
>    - Ideal for larger environments with tight chokepoints.
>    - Cleans at a rate of approximately 22,000 square feet per hour.
>    - Equipped with 40+ sensors for safety, providing a 360° view to handle unexpected obstacles.
>    - Features an easy-to-use touchscreen for operation and convenient dirty water disposal.
> 
> Both models provide excellent consistency and professional-grade cleaning. If you need more specific details or wish to book a demo, feel free to let me know!
> 
> **USER**: DUST-E SX seems more aligned with our needs, can we book a demo?
>
> **ASSISTANT**: Certainly! You can book a demo for the DUST-E SX by visiting the following link: [Book a Demo](https://www.richtechrobotics.com/dust-e-mx#section_contact-form). If you need further assistance or have any questions, feel free to ask!
---

## Evaluation
For a task-oriented dialogue system, you could use the evaluation script to automatically generate synthetic conversations, extracting task completion metrics for evaluating the whole system. For more details, please refer to the [evaluation](../Evaluation/UserSimulator.md) documentation.
1. First, create an API for the Agent you built. It will start an API on the default port 8000.
    ```
    python model_api.py  --input-dir ./examples/customer_service
    ```

2. Then, start the evaluation process:
   ```
    python eval.py \
    --model_api http://127.0.0.1:8000/eval/chat \
    --config ./examples/customer_service_config.json \
    --documents_dir ./examples/customer_service \
    --output-dir ./examples/customer_service
    ```

## Evaluation Results
The evaluation will generate the following outputs in the specified output directory:
1. **Simulated Synthetic Dataset (`simulate_data.json`)**  
   - JSON file containing simulated conversations generated based on the user's objective to evaluate the task success rate.
  
2. **Labeled Synthetic Dataset (`labeled_data.json`)**  
   - JSON file containing labeled conversations generated based on the taskgraph to evaluate the NLU performance.

3. **Goal Completion Metrics (`goal_completion.json`)**  
   - JSON file summarizing task completion statistics based on the bot's ability to achieve specified goals.