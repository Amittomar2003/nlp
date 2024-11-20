import streamlit as st
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled, NoTranscriptFound
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from transformers import pipeline
from textblob import TextBlob
import re
import nltk

# Ensure that necessary NLTK data is downloaded
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Function to preprocess text
def preprocess_text(text):
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()
    words = word_tokenize(text)
    cleaned_text = ' '.join(
        [lemmatizer.lemmatize(word.lower()) for word in words if word.isalnum() and word not in stop_words]
    )
    return cleaned_text

# Function to summarize text
def summarize_text(text, chunk_size=500, max_length=100, min_length=50):
    summarization_pipeline = pipeline("summarization")
    chunks = [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]
    summaries = [
        summarization_pipeline(chunk, max_length=max_length, min_length=min_length, do_sample=False)[0]['summary_text']
        for chunk in chunks
    ]
    return ' '.join(summaries)

# Function to extract keywords
def extract_keywords(text):
    processed_text = preprocess_text(text)
    vectorizer = CountVectorizer(max_features=10).fit([processed_text])
    return list(vectorizer.get_feature_names_out())

# Function to perform topic modeling (LDA)
def topic_modeling(text, n_topics=5):
    processed_text = preprocess_text(text)
    vectorizer = CountVectorizer(stop_words='english')
    tf = vectorizer.fit_transform([processed_text])
    lda_model = LatentDirichletAllocation(n_components=n_topics, max_iter=5, random_state=42)
    lda_model.fit(tf)
    feature_names = vectorizer.get_feature_names_out()
    topics = []
    for topic_idx, topic in enumerate(lda_model.components_):
        topics.append([feature_names[i] for i in topic.argsort()[:-6:-1]])
    return topics

# Function to extract YouTube video ID from URL
def extract_video_id(url):
    video_id = None
    patterns = [
        r'v=([^&]+)',  # Pattern for URLs with 'v=' parameter
        r'youtu.be/([^?]+)',  # Pattern for shortened URLs
        r'youtube.com/embed/([^?]+)'  # Pattern for embed URLs
    ]
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            video_id = match.group(1)
            break
    return video_id

# Main Streamlit app
def main():
    st.title("YouTube Video Summarizer")

    # User input for YouTube video URL
    video_url = st.text_input("Enter YouTube Video URL:", "")

    # User customization options
    max_summary_length = st.slider("Max Summary Length (per chunk):", 50, 1000, 200)

    if st.button("Summarize"):
        try:
            # Extract video ID from URL
            video_id = extract_video_id(video_url)
            if not video_id:
                st.error("Invalid YouTube URL. Please enter a valid URL.")
                return

            # Get transcript of the video
            transcript = YouTubeTranscriptApi.get_transcript(video_id)
            if not transcript:
                st.error("Transcript not available for this video.")
                return

            video_text = ' '.join([line['text'] for line in transcript])

            # Summarize the transcript
            summary = summarize_text(video_text, chunk_size=500, max_length=max_summary_length)

            # Extract keywords from the transcript
            keywords = extract_keywords(video_text)

            # Perform topic modeling
            topics = topic_modeling(video_text)

            # Perform sentiment analysis
            sentiment = TextBlob(video_text).sentiment

            # Display summarized text, keywords, topics, and sentiment
            st.subheader("Video Summary:")
            st.write(summary)

            st.subheader("Keywords:")
            st.write(keywords)

            st.subheader("Topics:")
            for idx, topic in enumerate(topics):
                st.write(f"Topic {idx+1}: {', '.join(topic)}")

            st.subheader("Sentiment Analysis:")
            st.write(f"Polarity: {sentiment.polarity}")
            st.write(f"Subjectivity: {sentiment.subjectivity}")

        except TranscriptsDisabled:
            st.error("Transcripts are disabled for this video.")
        except NoTranscriptFound:
            st.error("No transcript found for this video.")
        except Exception as e:
            st.error(f"Error: {str(e)}")

if __name__ == "__main__":
    main()


