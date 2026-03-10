# From Idea to AI: Building Applications with Generative AI

## Introduction

Just the other day, I noticed that Gartner reported that 80% of enterprises will have used some type of generative AI through models or APIs by 2026.

As a developer, this might seem daunting if you've only used AI through co-pilots in your IDE or popular large language models online, but haven't built applications with AI. The good news is that it's easier than you might think to get started with AI as a developer.

Today we're focused on:
- Where to get started with AI
- How to build AI-powered applications
- Where to run them
- Open source tools and technologies that can help in building, running, and testing applications with AI

## The Three Main Steps of the AI Journey

When going from a simple proof of concept to a production application, developers experience three main phases:

1. **Ideation and Experimentation**
2. **Building**
3. **Development and Operations**

## Phase 1: Ideation and Exploration

### Getting Started

How do you get started building an application that uses generative AI? The first step is ideating around exploration and proof of concepts. Here's how to break it down:

**Step 1: Start with a Specialized Model**

Remember that your use case is specialized, so you need a specialized model that can do the job well. Research and evaluate models from popular repositories like:
- Hugging Face
- Open source communities

### Understanding Key Factors

You need to consider different factors such as:
- Model size
- Performance
- Benchmarking through popular benchmark tools

### Important Ground Rules

**Self-Hosting vs. Cloud-Based:**
- Self-hosting a large language model (LLM) is generally cheaper than a cloud-based service

**Model Types:**
- Small Language Models (SLMs) vs. Large Language Models (LLMs)
  - SLMs/LLMs generally perform better with lower latency
  - They are specialized for specific tasks

### Prompting Techniques

When working with models, understand various prompting techniques:

**Zero-Shot Prompting**
- Asking a model a question without any examples of how to respond

**Few-Shot Prompting**
- Providing a few different examples of how you want the LLM to respond
- Establishing the behavior you want the AI to have

**Chain of Thought**
- Asking the model to explain its thinking and process step by step

### Experimental Validation

You need to understand the different capabilities and limitations of the models you're working with. Experiment with your data early on to understand any potential challenges that might come up as you go through the AI journey.

## Phase 2: Building AI Applications

### Local Development Setup

Just as you can locally run databases and different services on your machine, you can do the same with AI:
- Serve AI locally from your machine
- Make requests to its API from localhost
- Ensure your data is secure and private on-premise

This is really important for data privacy nowadays.

### Methods to Incorporate Your Data

#### Retrieval-Augmented Generation (RAG)

Take a large language model (pre-trained foundational model) and supplement it with relevant and accurate data. This can help provide better and more accurate responses.

#### Fine-Tuning

Take the large language model and include your data with it. You're baking in:
- The information you want it to know
- How you want it to behave
- The different styles and intuitions you want it to react with

Then you can inference it and have that domain-specific data every time you work with the AI model.

### Tools and Frameworks

Having the right tools and frameworks, such as **LangChain**, simplifies your life. These tools:
- Let you focus on building new features
- Support popular GenAI use cases like:
  - Chatbots
  - IT process automation
  - Data management
  - And much more
- Simplify the different calls you'll make through the model

### Building Complex Tasks

You can accomplish more complex tasks through sequences of prompts and model calls. This means you need to:
- Break down problems into smaller, more manageable steps
- Evaluate the flows during model calls in both development and production environments

## Phase 3: Development and Operations (MLOps)

### Deploying to Production

Once you've built your AI-powered application, you'll want to deploy it to production to scale things up. This falls under the umbrella of **Machine Learning Operations (MLOps)**.

### Key Infrastructure Considerations

Your infrastructure needs to handle:
- **Efficient Model Deployment and Scaling**
  - Use containers and orchestrators like **Kubernetes**
  - Auto-scale and balance traffic for your application
- **Production-Ready Runtime**
  - Use production-ready services like **vLLM** for model serving

### Hybrid Approach

Organizations are increasingly taking a hybrid approach with:
- **Multi-model Strategy**: A "Swiss Army knife" approach with different models for different use cases
- **Hybrid Infrastructure**: A combination of on-premise and cloud infrastructure to make the most of your resources and budget

### Ongoing Monitoring and Management

The job doesn't end at deployment. You still need to:
- Benchmark your models
- Monitor performance
- Handle exceptions from your application
- Ensure models go into production smoothly (similar to DevOps)

## Conclusion

Recent innovations in the world of AI have made this topic much more accessible for developers. You have plenty of tools available to help you along the process.

The key takeaway: **AI is just another tool that you can add to your tool belt.**

Use these tools and follow the process to go from ideation, to building, to deployment of AI-powered applications and make a real impact with your work using GenAI.
