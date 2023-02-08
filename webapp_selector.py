import streamlit as st
import requests
import arxiv
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lex_rank import LexRankSummarizer

nltk.download('wordnet')
nltk.download('punkt')
headers = {"Authorization": f"Bearer {st.secrets['API_KEY']}",
	"content-type": "application/json"}
    
#==============================================================================================================
# Streamlit UI
st.set_page_config(page_title="Arxiv Summary Generator", 
                   page_icon=":newspaper:",
                   layout="wide")
st.title("Arxiv Summary Generator")
st.markdown("Please paste an [Arxiv]( https://arxiv.org ) paper link below and select type of AI/Python model for writing summary or [Contact us](info@chatgptai.me)")
# CSS
#==============================================================================================================
def summarize_sum(text, ratio=0.5):
    parser = PlaintextParser.from_string(text, Tokenizer("english"))
    summarizer = LexRankSummarizer()

    summary = summarizer(parser.document, ratio * len(parser.document.sentences))

    return " ".join([str(sentence) for sentence in summary])

def summarize_nltk(text, ratio=0.5):
    stop_words = set([word.strip() for word in open("english_stopwords.txt","r",encoding="utf-8").read().split("\n")])
    words = word_tokenize(text)
    words = [word.lower() for word in words if word.isalpha() and word.lower()
not in stop_words]
    sentences = sent_tokenize(text)

    word_frequencies = {}
    for word in words:
        if word not in word_frequencies.keys():
            word_frequencies[word] = 0
        word_frequencies[word] += 1

    maximum_frequency = max(word_frequencies.values())
    for word in word_frequencies.keys():
        word_frequencies[word] = (word_frequencies[word]/maximum_frequency)

    sentence_scores = {}
    for i, sentence in enumerate(sentences):
        for word in nltk.word_tokenize(sentence.lower()):
            if word in word_frequencies.keys():
                if i not in sentence_scores.keys():
                    sentence_scores[i] = word_frequencies[word]
                else:
                    sentence_scores[i] += word_frequencies[word]

    summary_sentences = sorted(sentence_scores.items(), key=lambda x: x[1],
reverse=True)
    summary_sentences = summary_sentences[:int(len(summary_sentences)*ratio)]
    summary_sentences.sort()

    summary = " ".join([sentences[index] for (index, score) in
summary_sentences])
    return summary

# Function to fetch the abstract of the arxiv paper
def get_arxiv_info(url):
    #https://arxiv.org/abs/2301.00029/
    id = url.split("/")[-1].replace("/","")
    search = arxiv.Search(id_list=[id])
    return next(search.results())
dictList = lambda x,y:[y[i] for i in sorted(x)]


def openai(abstract, model="text-curie-001",
                length=100,
                temperature=0.7,
                top_k=0,
                top_p=0.9,
                num_return_sequences=1, TRANSLATE = ""):
    #API_KEY = 
    prompt =  f"Summarize following text: {abstract}"
    request_body =  {"model": "text-curie-001", 
                    "prompt": prompt,
                    "max_tokens": length, 
                    "temperature": temperature, 
                    "frequency_penalty": top_k, 
                    "presence_penalty": top_p,
                    "n": num_return_sequences}
    response = requests.post("https://api.openai.com/v1/completions", json=request_body, headers=headers)
    data = response.json()
    #print(data)
    summary_text = data["choices"][0]["text"]
    tokens = f"Prompt Tokens: {data['usage']['prompt_tokens']},\nCompletion Tokens: {data['usage']['completion_tokens']}, \nTotal Tokens: {data['usage']['total_tokens']}"
    return summary_text, tokens
#==============================================================================================================
# START FROM HERE       
url = st.text_input("Enter the arXiv URL", "https://arxiv.org/abs/2102.01755")
selected_function = st.selectbox("Summary Type: ", ["OpenAI-Davinci-LITE", "OpenAI-Curie-LITE",
                                                    "OpenAI-Davinci", "OpenAI-Curie",
                                                    "NLTK", "Python_Summarizer_Sumy"])
fetch_button = st.button("Fetch")
# Create a dropdown selector
#==============================================================================================================

#==============================================================================================================
if fetch_button:
    #st.write("Fetching the abstract...")
    with st.spinner("Processing..."):
        paper = get_arxiv_info(url)

        
        st.header(paper.title)
        #print([i.__dict__ for i in paper.authors])
        x = ":".join([author.name for author in paper.authors])
        st.header(f"_{x}_")
        st.write(f"**Published On:**  _{paper.published}_")
        st.write("**Abstract:**")
        st.write(paper.summary)
        st.write("---")
        # Call the selected function
        if selected_function == "NLTK":
            st.write("**[NLTK] Summary:**")
            nltkS = summarize_nltk(paper.summary)
            st.write(nltkS)
            st.write("---")

        elif selected_function == "Python_Summarizer_Sumy":
            st.write("**[Python-SUMY] Summary:**")
            sumyS = summarize_sum(paper.summary)
            st.write(sumyS)
            st.write("---")

        elif selected_function == "OpenAI-Curie":
            st.write("**Generated using OpenAI's GPT-3**")
            abstract = paper.summary
            ai_summary1, tokens1 = openai(paper.summary, 
                                        model="text-curie-001", 
                                        length=200, 
                                        temperature=0.7, 
                                        top_k=1, 
                                        top_p=1, 
                                        num_return_sequences=1)
            st.write("**AI-Summary: [text-curie-001]**")
            st.write(ai_summary1)
            st.write("---")
            st.write("**TOKEN INFO: [text-curie-001]**")
            for i in tokens1.split(","): st.write(i)
            st.write("---")
        elif selected_function == "OpenAI-Davinci":
            st.write("_Generated using OpenAI's GPT-3_")
            abstract = paper.summary
            ai_summary2, tokens2 = openai(paper.summary, 
                                        model="text-davinci-003", 
                                        length=200, 
                                        temperature=0.7, 
                                        top_k=1, 
                                        top_p=1, 
                                        num_return_sequences=1)
            st.write("**AI-Summary: [text-davinci-003]**")
            st.write(ai_summary2)
            st.write("---")
            st.write("**TOKEN INFO: [text-davinci-003]**")
            for i in tokens2.split(","): st.write(i)
            st.write("---")
                
        elif selected_function == "OpenAI-Curie-LITE":
            st.write("**Generated using OpenAI's GPT-3**")
            ai_summary1, tokens1 = openai(url, 
                                        model="text-curie-001", 
                                        length=200, 
                                        temperature=0.7, 
                                        top_k=1, 
                                        top_p=1, 
                                        num_return_sequences=1)
            st.write("*AI-Summary: [text-curie-001]-LITE*")
            st.write(ai_summary1)
            st.write("---")
            st.write("*TOKEN INFO: [text-curie-001]-LITE*")
            for i in tokens1.split(","): st.write(i)
            st.write("---")
        elif selected_function == "OpenAI-Davinci-LITE":
            st.write("**_Generated using OpenAI's GPT-3_**")
            abstract = paper.summary
            ai_summary2, tokens2 = openai(url, 
                                        model="text-davinci-003", 
                                        length=200, 
                                        temperature=0.7, 
                                        top_k=1, 
                                        top_p=1, 
                                        num_return_sequences=1)
            st.write("**AI-Summary: [text-davinci-003]-LITE**")
            st.write(ai_summary2)
            st.write("---")
            st.write("**TOKEN INFO: [text-davinci-003]-LITE**")
            for i in tokens2.split(","): st.write(i)
            st.write("---")
#==============================================================================================================
                
