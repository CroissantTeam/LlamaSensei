from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy
from datasets import Dataset
from langchain_groq import ChatGroq
from langchain_community.embeddings import SentenceTransformerEmbeddings


# Load the embedding model
model = ChatGroq(model="llama3-70b-8192",temperature=0)
EMBEDDING_LLM = SentenceTransformerEmbeddings(model_name = "all-MiniLM-L12-v2", model_kwargs={"trust_remote_code":True})

# Step 1: Define the Functions

def calculate_faithfulness(query: str, generated_answer: str, contexts: list) -> float:
    # Prepare the dataset for evaluation
    data_samples = {
        'question': [query],
        'answer': [generated_answer],
        'contexts': [contexts],
    }
    dataset = Dataset.from_dict(data_samples)

    # Evaluate faithfulness
    score = evaluate(dataset, metrics=[faithfulness], llm = model, embeddings=EMBEDDING_LLM)

    # Convert score to pandas DataFrame and extract the faithfulness score
    score_df = score.to_pandas()
    faithfulness_score = score_df['faithfulness'].iloc[0]
    return faithfulness_score

def calculate_answer_relevancy(query: str, generated_answer: str, contexts: list) -> float:
    # Prepare the dataset for evaluation
    data_samples = {
        'question': [query],
        'answer': [generated_answer],
        'contexts': [contexts],
    }
    dataset = Dataset.from_dict(data_samples)

    # Evaluate answer relevancy
    score = evaluate(dataset, metrics=[answer_relevancy], llm = model, embeddings=EMBEDDING_LLM)

    # Convert score to pandas DataFrame and extract the answer relevancy score
    score_df = score.to_pandas()
    answer_relevancy_score = score_df['answer_relevancy'].iloc[0]
    return answer_relevancy_score

# Step 2: Create a Toy Dataset

toy_dataset = {
    "question": [
        "What method do we use if we want to predict house price in an area?",
    ],
    "answer": [
        "To predict house prices in an area, you would typically use a supervised learning approach, specifically a regression model. "
        "A common method is to fit a straight line (linear regression) to the dataset, where the inputs (such as the size of the house, "
        "number of bedrooms, ZIP code, and wealth of the neighborhood) are used to predict the continuous output, which is the house price. "
        "In some cases, more complex models like quadratic functions or neural networks with appropriate activation functions (such as ReLU) "
        "are used."
    ],
    "contexts": [
        [
            """I guess technically prices can be rounded off to the nearest dollar and cents. So prices aren't really real numbers. 
            You know, but because you probably not price it. How's that like Pi times 1,000,000 or whatever? But so- so- but- but- but for all practical purposes, 
            prices are continuous. So we call them housing price prediction to be a regression problem. Whereas if you have, two values, a possible output is 0 or 1, 
            we call that a classification problem. If you have k discrete outputs, so, if the tumor can be, malignant, or if there are 5 types of cancer, right, 
            so you have 1 of 5 possible outputs, then that's also a classification problem if the output is discrete. Now, I wanna find a different way to visualize 
            this dataset which is, let me draw a line on top, and I'm just gonna, you know, map all this data on the horizontal axis upward onto a line. 
            But, well, let me show you what I'm gonna do. I'm going to use a symbol o to denote, right? I hope what I did was clear. So I took the 2 sets of examples, 
            the positive and negative examples. Positive examples is 1, negative examples is 0. And I took all of these examples and- and kinda pushed them up onto a 
            straight line, and I used 2 symbols.""",
            
            """This is data from, Portland, Oregon. But so there's the size of a house in square feet, and that's the price of a house in 1,000 of dollars. 
            Right? And so there's a house that is, 2,104 square feet whose asking price was $400,000, house with, that size with that price. And so on. Okay? 
            And maybe more conventionally, if you plot this data, there's the size, there's the price. So you have some data set like that, and what we'll end up 
            doing today is, fit a straight line to this data, right, and go through how to do that. So in supervised learning, the process of supervised learning is 
            that you have a training set, such as the dataset that I drew on the left, and you feed this to a learning algorithm. Right. And the job of the learning 
            algorithm is to output a function, to make predictions about housing prices. And by convention, I'm gonna call this function that it outputs a hypothesis. 
            Right. And the job of the hypothesis is, you know, it will, it can input the size of a new house, the size of a different house that you haven't seen yet, 
            and will output the estimated price. Okay?""",
            
            """Right? And you want to know, you know, how do you price this house? So given this dataset, one thing you can do is, fit a straight line to it. 
            Right? And then you could estimate or predict the price to be whatever value you read off on the, vertical axis. So in supervised learning, you are 
            given a dataset with, inputs x and labels y, and your goal is to learn a mapping from x to y. Right? Now, fitting a straight line to data is maybe the 
            simplest possible- maybe the simplest possible learning algorithm, maybe the, one of the simplest learning algorithms. Given the data set like this, 
            there are many possible ways to learn the mapping, to learn the function, mapping from the input size to the estimated price. And so, maybe you wanna 
            fit a quadratic function instead. Maybe that actually fits the data a little bit better. And so how do you choose among different models will be, either 
            automatically or manual intervention will be- will be something we'll spend a lot of time talking about. Now to give a little bit more, to define a few 
            more things, this particular example is a problem called a regression problem. And the term regression refers to that the, value y you're trying to predict 
            is continuous, right? In contrast, here is a- here's a different type of problem.""",
            
            """Okay. So we will see why we use ReLU mostly. Yeah. Yeah. For example, you remember the house prediction example? In that case, if you wanna if you wanna 
            predict the price of a house based on some features, you would use ReLU because you know that the output should be a positive number between 0 and plus 
            infinity. It doesn't make sense to use one of tanh or sigmoid. Yeah. When would you, like, how would you differentiate between using tanh and sigmoid? 
            Doesn't really matter. I think if if I want my output to be between 0 and 1, I would use sigmoid. If I want my output to be between minus 1 and 1, I would 
            use tanh. So, you know, there is there is some tasks where the output is kind of a reward or a minus reward that you wanna get. Like in reinforcement learning, 
            you would use tanh as an output activation which is because minus 1 looks like a negative reward, plus 1 looks like a positive reward, and you wanna decide 
            what should be the reward. Good question.""",
            
            """Let me give you another example which is a house prediction example. House price prediction. So let's assume that our inputs are number of bedrooms, 
            size of the house, ZIP code, and wealth of the neighborhood, let's say. What we will build is a network that has 3 neurons in the first layer and 1 neuron 
            in the output layer. So what's interesting is that as a human, if you were to build, this network and, like, hand engineer it, you would say that, okay. 
            ZIP codes and wealth are are sorry. Never mind. Let's do that. ZIP code and wealth are able to tell us about the school quality in the neighborhood, 
            the quality of the school that is next to the house, probably. As a human, you would say these are probably good features to predict that. The ZIP code 
            is going to tell us if the neighborhood is walkable or not, probably. The size and the number of bedrooms is going to tell us what's the size of the 
            family that can fit in this house. And these 3 are probably better information than these in order to finally predict the price. So that's a way to hand 
            engineer that by hand as a human in order to give human knowledge to the network to figure out the price. In practice, what we do here is that we use a 
            fully connected layer."""
        ]
    ]
}

# Step 3: Run the Toy Example

# Example usage for the "Solar System" question
query = toy_dataset["question"][0]
generated_answer = toy_dataset["answer"][0]
contexts = toy_dataset["contexts"][0]

# Calculate faithfulness
faithfulness_score = calculate_faithfulness(query, generated_answer, contexts)
print(f"Faithfulness Score: {faithfulness_score}")

# Calculate answer relevancy
answer_relevancy_score = calculate_answer_relevancy(query, generated_answer, contexts)
print(f"Answer Relevancy Score: {answer_relevancy_score}")
