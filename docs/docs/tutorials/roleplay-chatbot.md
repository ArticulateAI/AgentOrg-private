# Roleplay Bot

*Build a basic chatbot with MessageWorkers*

## Intro

In this tutorial, we'll walk through building a roleplay bot using **Agent First Organization**'s framework. This bot will pretend to be a role you assigned to it and have a practice conversation with you based on the objective you set.

By the end of this tutorial, you'll know how to set up the config file, build a basic conversation flow, and power a bot with it! This tutorial covers the absolute basics of our framework and sets up for much more!

## Setting up the Config File

In its core, bot is powered through a [TaskGraph](../Taskgraph/intro.md) which is the structure that links various tasks together to fulfill the overall role of the bot. Every "node" represents a task which has an [Worker](../Workers/Workers.md) that is selected to complete task, and would engages the user for their response. With the user response, the TaskGraph will decide which next node to travel to.

While there are a lot of pre-built workers (and the ability to write custom workers too!), for this simple task, we will only be using two of the most basic workers. [MessageWorker](../Workers/MessageWorker.mdx) and [DefaultWorker](../Workers/DefaultWorker.mdx). Message Worker is the base worker in handling message responses and DefaultWorker is there in case MessageWorker is not deemed suitable. 

Here is the simple structure for a [Config](../Config.md) JSON file:

* `role (Required)`: The general "role" of the chatbot you want to create. For instance, "roleplay bot", "customer service assistant", "data analyst", "shopping assistant", etc.
* `user_objective (Required)`: The user's goal that the chatbot wants to achieve. Related to the user experience. Description in third person. For instance, "The customer service assistant helps users with customer service inquiries. It can provide information about products, services, and policies, as well as help users resolve issues and complete transactions."
* `workers (Required, List(WorkerClassName))`: The [Workers](Workers/Workers.md) pre-defined under `agentorg/workers` folder in the codebase that you want to use for the chatbot.

Now, lets see it with the Roleplay Bot example.

```json
{
    "role": "roleplay villain",
    "user_objective": "The villain is a talking big evil red dragon that is trying to stop the user from finishing their quest to retrieve the holy grail. It is deadly and have killed many knights and heros in the past.",
    "builder_objective": "",
    "domain": "action roleplay",
    "intro": "",
    "docs": [],
    "tasks": [],
    "workers": [
        "MessageWorker"
    ]
}
```

With our Config in place, the vast majority of work is surprisingly already done! The rest is simply bringing the bot to life.

## Generating a TaskGraph

Now that we have a Config file, generating the graph is the easy part. All you need to do is run the following command:

`python create.py --config ./examples/roleplay.json --output-dir ./examples/roleplay`

It will first enter into a *task planning* interactive panel where you can see the generated tasks this bot will handle and the following steps to complete the specific tasks. You can also modify the tasks and steps as needed. Once you are satisfied with result, you could press `s` to save the *task planning* file then it will further generate the final *TaskGraph* file. 

TaskGraph provides the graph that the bot will traverse through during the conversation. It provides a guideline for the conversation to make it more controllable and reliable. The details can be viewed at [here](../Taskgraph/Generation.md). 

## Running the Bot

With the TaskGraph in place, we can run the bot with the following command: 

`python run.py --input-dir ./examples/roleplay`

It will initialize the service (e.g. NLU apis) you need to start the bot and you can start interacting with it!

---

## Sample Conversation
Here is a sample conversation of the roleplay bot:
```
> Bot: Greetings, foolish adventurer! I am the mighty red dragon, guardian of the secrets you seek. Turn back now, lest you join the countless knights who have perished before you. What brings you to my fiery domain in pursuit of the elusive holy grail?

> User: I am questing for the holy grail for the hand of the princess!

> Bot: Greetings, intrepid seeker! How may I be of service to you today? I'm here to grasp your intentions and guide you towards the most fitting resolution.

> User: I want the grail.

> Bot: Ah, you dare to claim the grail, do you? Know this, brave soul: many have tried and met their doom at my claws. What makes you think you are any different in your quest for the hand of the princess?

> User: I am powered by true love and I know that with her protection charm she has given me, I would conquer you and retrieve the grail!   

> Bot: Ah, you dare to claim the grail, do you? Know this, brave soul: many have tried and met their doom at my claws. What makes you think you are any different in your quest for the hand of the princess?

> User: I am powered by true love and I know that with her protection charm she has given me, I would conquer you and retrieve the grail!   

> Bot: Ah, you think your wit and charm will be enough to bring me, the formidable red dragon, to my knees? Many have tried to outsmart and sway me with their clever words, only to meet their end in my fiery domain. What makes you so confident that your quest for the holy grail and the princess's hand will lead to a different fate?
```