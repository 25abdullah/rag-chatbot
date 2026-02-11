from fastapi import FastAPI, WebSocket, HTTPException, UploadFile
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from supabase import create_client
from openai import OpenAI
from datetime import datetime
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
import os
import aiofiles
from pathlib import Path
import uuid
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from dotenv import load_dotenv
import os
from transformers import (
    VisionEncoderDecoderModel,
    TrOCRProcessor,
    BlipProcessor,
    BlipForConditionalGeneration,
)
from PIL import Image


# initialize app and set up cors
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
supabase_client = create_client(SUPABASE_URL, SUPABASE_KEY)
OPENROUTER_KEY = os.getenv("OPENROUTER_API_KEY")

ai_model = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=OPENROUTER_KEY)

embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")


# TrOCR for text extraction from images
ocr_processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-handwritten")
ocr_model = VisionEncoderDecoderModel.from_pretrained(
    "microsoft/trocr-base-handwritten"
)

# Image captioning for photo descriptions
caption_processor = BlipProcessor.from_pretrained(
    "Salesforce/blip-image-captioning-large"
)
caption_model = BlipForConditionalGeneration.from_pretrained(
    "Salesforce/blip-image-captioning-large"
)


CONV_RETRIEVAL_COUNT = 5
GLOBAL_RETRIEVAL_COUNT = 3
CHUNK_FREQUENCY = 8
UPLOAD_DIR = "uploaded_files"


os.makedirs(UPLOAD_DIR, exist_ok=True)


app.mount("/static", StaticFiles(directory="static"), name="static")


SYSTEM_PROMPT = """
You are an intelligent, personalized AI assistant with exceptional 
memory capabilities. You maintain context across conversations and use 
past interactions to provide thoughtful, relevant responses.

--- MEMORY & CONTEXT USAGE ---
You have access to information from previous conversations with this 
user AND from any files they've uploaded. This information appears 
below in the "WHAT YOU REMEMBER" section.

**CRITICAL RULES:**
1. **USE the provided context** - If information is in "WHAT YOU 
   REMEMBER", you KNOW it. Don't claim ignorance.
2. **Be specific** - Reference memories directly: "You mentioned you 
   love basketball" NOT "Based on the context..."
3. **Be natural** - Incorporate memories seamlessly
4. **Don't over-explain** - Avoid meta-commentary like "I see in my 
   records that..."
5. **Prioritize recent context** - More recent information is likely 
   more relevant

--- HANDLING FILES & DOCUMENTS ---
When you retrieve content from uploaded files:
- USE IT directly in your response
- NEVER say "I don't have access to files" if content appears below
- Reference the content naturally: "According to the document..." or 
  "The file shows..."
- If asked about a file, summarize or answer based on the retrieved 
  content

--- WHEN USER ASKS ABOUT THEMSELVES ---
If the user asks "what do you know about me?", "tell me about 
myself", or similar:
- Share relevant details from "WHAT YOU REMEMBER" section
- Be organized and clear
- Group related information
- If no context exists, say: "We haven't chatted much yet, so I 
  don't know much about you. Tell me about yourself!"

--- HANDLING VAGUE QUERIES ---
For vague queries like "tell me more" or "what else":
- Use conversation context to infer what they're asking about
- Reference specific details from memory when relevant
- Ask clarifying questions if truly ambiguous

--- PERSONAL DETAILS TO REMEMBER ---
Pay special attention to:
- **Name** - Use it naturally when appropriate
- **Interests & hobbies** - Sports, activities, pastimes
- **Work/Study** - Job, major, university, projects
- **Preferences** - Favorite things
- **Goals & plans** - What they're working toward
- **Relationships** - People they mention
- **Locations** - Where they live, study, work

--- RESPONSE STYLE ---
- Be conversational and warm
- Keep responses concise unless detail is requested
- Use the user's name occasionally (not every message)
- Match the user's tone (formal vs casual)
- Never be condescending or overly explanatory

--- HANDLING CONTRADICTIONS ---
If new information contradicts old memories:
- Prioritize the NEW information (people change!)
- Don't point out the contradiction unless asked
- Update your understanding naturally

--- WHAT YOU REMEMBER ABOUT THIS USER ---
{memory_text}

--- RESPONSE INSTRUCTIONS ---
Now respond to the user's message naturally, incorporating relevant 
memories or file content when appropriate. Be helpful, personalized, 
and conversational.
"""


# conversation -> voice
# Given a converastion id, allows for users to chat with AI with streaming
@app.websocket("/conversations/{conversation_id}/chat")
async def chat_websocket(websocket: WebSocket, conversation_id: str):
    await websocket.accept()  # get connection
    try:
        while True:
            user_message = await websocket.receive_text()

            memory_text = retrieve_context(conversation_id, user_message)

            system_prompt = SYSTEM_PROMPT.format(memory_text=memory_text)

            ai_response = await generate_streaming_response(
                system_prompt, user_message, websocket
            )
            store_messages(conversation_id, user_message, ai_response)

            embed_messages(conversation_id, user_message, ai_response)
    except Exception as e:
        print(f"Exception {e}")


def retrieve_context(conversation_id, query):
    """
    This function retrives 5 different pieces of context: same conversation messages, same conversation files, other conversation
    messages (excluding current conversation), other conversation files (excluding current), and chunks of other conversations (excluding this converesation).


    :param conversation_id: conversation identifer for current conversation
    :param query: the user message that is used to find the most relevant information.
    """
    same_conversation_messages = ""
    same_conversation_files = ""
    other_conversation_messages = ""
    other_conversation_chunks = ""
    other_conversation_files = ""

    # 1. Messages from this conversation
    try:
        results = get_conversation_collection(conversation_id).similarity_search(
            query, k=5, filter={"type": "message"}
        )
        for res in results:
            same_conversation_messages += res.page_content + "\n"
        print(f"[1] Conv messages: {len(results)} results")
    except:
        same_conversation_messages = ""

    # 2. Files from this conversation
    try:
        results = get_conversation_collection(conversation_id).similarity_search(
            query, k=3, filter={"type": "file"}
        )
        for res in results:
            same_conversation_files += res.page_content + "\n"
        print(f"[2] Conv files: {len(results)} results")

    except:
        same_conversation_files = ""

    # 3. Messages from other conversations
    try:
        results = get_global_collection().similarity_search(
            query,
            k=3,
            filter={
                "$and": [
                    {"type": "message"},
                    {"conversation_id": {"$ne": conversation_id}},
                ]
            },
        )
        for res in results:
            other_conversation_messages += res.page_content + "\n"
        print(f"[3] Global messages (other convs): {len(results)} results")
    except:
        other_conversation_messages = ""

    # 4. Chunks from other conversations
    try:
        results = get_global_collection().similarity_search(
            query,
            k=2,
            filter={
                "$and": [
                    {"type": "chunk"},
                    {"conversation_id": {"$ne": conversation_id}},
                ]
            },
        )
        for res in results:
            other_conversation_chunks += res.page_content + "\n"
        print(f"[4] Global chunks (other convs): {len(results)} results")
    except:
        other_conversation_chunks = ""

    # 5. Files from other conversations
    try:
        results = get_global_collection().similarity_search(
            query,
            k=2,
            filter={
                "$and": [
                    {"type": "file"},
                    {"conversation_id": {"$ne": conversation_id}},
                ]
            },
        )
        for res in results:
            other_conversation_files += res.page_content + "\n"
        print(f"[5] Global files (other convs): {len(results)} results")
    except:
        other_conversation_files = ""

    # Build structured context
    memory_text = ""

    if same_conversation_messages:
        memory_text += (
            "--- This Conversation Messages ---\n" + same_conversation_messages + "\n"
        )

    if other_conversation_messages or other_conversation_chunks:
        memory_text += (
            "--- Past Conversations Messages ---\n" + other_conversation_messages + "\n"
        )

    if other_conversation_chunks:
        memory_text += (
            "--- Global Conversation Chunks ---\n" + other_conversation_chunks + "\n"
        )

    if same_conversation_files:
        memory_text += (
            "--- Uploaded Documents From this Conversation ---\n"
            + same_conversation_files
            + "\n"
        )

    if other_conversation_files:
        memory_text += (
            "--- Uploaded Documents From Other Conversations ---\n"
            + other_conversation_files
            + "\n"
        )

    if not memory_text:
        memory_text = "No previous conversations yet. This is a fresh start!"
    print(f"=== FINAL CONTEXT ===\n{memory_text}\n=== END CONTEXT ===")
    return memory_text


async def generate_streaming_response(full_system_prompt, user_message, websocket):
    """
    The purpose of this function is to create the "streaming" response of LLM to the user.

    :param full_system_prompt: represents the system prompt (when this function is used, includes the RAG)
    :param user_message: the message that the user sent
    :param websocket: websocket connection
    """
    # make AI response
    response = ai_model.chat.completions.create(
        model="meta-llama/llama-3.3-70b-instruct:free",
        messages=[
            {"role": "system", "content": full_system_prompt},
            {"role": "user", "content": user_message},
        ],
        stream=True,
    )
    # build AI response
    build_response = ""
    for chunk in response:
        token = chunk.choices[0].delta.content
        if token:
            build_response += token
            await websocket.send_text(token)
    return build_response


def store_messages(conversation_id, user_msg, ai_msg):
    """
    The purpose of this function is to the store the AI and user message into supabase table.

    :param conversation_id: represents the current conversation identifier.
    :param user_msg: represents the message that the user sent.
    :param ai_msg: represents the ai's response to the user.
    """
    user_data_to_insert = {
        "conversation_id": conversation_id,
        "role": "user",
        "content": user_msg,
    }

    ai_data_to_insert = {
        "conversation_id": conversation_id,
        "role": "assistant",
        "content": ai_msg,
    }

    supabase_client.table("messages").insert(
        [user_data_to_insert, ai_data_to_insert]
    ).execute()


def embed_messages(conversation_id, user_msg, ai_msg):
    """
    This function embeds all the messages and stores them into the chromadb for rag.

    :param conversation_id: current conversation id
    :param user_msg: the message the user sent
    :param ai_msg: the ai's response to user's message
    """

    # Conversation collection
    get_conversation_collection(conversation_id).add_texts(
        texts=[user_msg],
        ids=[f"{conversation_id}_user_{datetime.now().timestamp()}"],
        metadatas=[
            {"type": "message", "role": "user", "conversation_id": conversation_id}
        ],
    )
    get_conversation_collection(conversation_id).add_texts(
        texts=[ai_msg],
        ids=[f"{conversation_id}_ai_{datetime.now().timestamp()}"],
        metadatas=[
            {"type": "message", "role": "assistant", "conversation_id": conversation_id}
        ],
    )

    # Global collection — individual messages for immediate cross-conversation recall
    get_global_collection().add_texts(
        texts=[user_msg],
        ids=[f"global_{conversation_id}_user_{datetime.now().timestamp()}"],
        metadatas=[
            {"type": "message", "role": "user", "conversation_id": conversation_id}
        ],
    )
    get_global_collection().add_texts(
        texts=[ai_msg],
        ids=[f"global_{conversation_id}_ai_{datetime.now().timestamp()}"],
        metadatas=[
            {"type": "message", "role": "assistant", "conversation_id": conversation_id}
        ],
    )
    print(f"Embedded to conv + global for conversation: {conversation_id}")
    chunk_to_global(conversation_id)


# return full conversation given an id for chroma side (not supabase)
def get_conversation_collection(conversation_id):
    return Chroma(
        persist_directory="./chroma_db",
        embedding_function=embeddings,
        collection_name=f"conv_{conversation_id}",
    )


# get full collection of all conversations
def get_global_collection():
    return Chroma(
        persist_directory="./chroma_db",
        embedding_function=embeddings,
        collection_name="user_123_global",
    )


def chunk_to_global(conversation_id):
    """
    The purpose of this function is to store a large number messages (8 or so)
    in a row and then embed all of that to provide the LLM greater context for when it
    uses RAG.

    :param conversation_id: represents the current conversation id
    """
    try:
        print(f"Checking if should chunk for conversation: {conversation_id}")
        response = (
            supabase_client.table("messages")
            .select("*", count="exact")
            .eq("conversation_id", conversation_id)
            .execute()
        )
        print(f"Message count: {response.count}")
        print(f"Count % {CHUNK_FREQUENCY} = {response.count % CHUNK_FREQUENCY}")

        if response.count % CHUNK_FREQUENCY == 0:
            print(f"Chunking to global! Message count: {response.count}")
            get_last_ten = (
                supabase_client.table("messages")
                .select("*")
                .eq("conversation_id", conversation_id)
                .order("created_at", desc=True)
                .limit(CHUNK_FREQUENCY)
                .execute()
            )
            group_messages = ""
            rows = get_last_ten.data
            for row in rows:
                group_messages += row["content"]
            get_global_collection().add_texts(
                texts=[group_messages],
                ids=[f"chunk_{conversation_id}_{datetime.now().timestamp()}"],
                metadatas=[{"type": "chunk", "conversation_id": conversation_id}],
            )
            print("Successfully chunked to global!")
    except Exception as e:
        print(f"Exception in chunk_to_global: {e}")


# create a conversation
@app.post("/conversations")
async def create_conversation():
    try:
        response = supabase_client.table("conversations").insert({}).execute()
        created_conversation = response.data[0]
        return created_conversation
    except Exception as e:
        print(f"Exception {e}")


# delete conversation and corresponding data
@app.delete("/conversations/{conversation_id}")
async def delete_conversation(conversation_id: str):
    try:
        supabase_client.from_("messages").delete().eq(
            "conversation_id", conversation_id
        ).execute()
        supabase_client.table("conversations").delete().eq(
            "id", conversation_id
        ).execute()
        return {
            "status": "success",
            "message": f"Deleted conversation {conversation_id}",
        }
    except Exception as e:
        print(f"Error deleting conversation id: {conversation_id}")
        raise HTTPException(status_code=500, detail="Failed to delete conversation.")


# return all conversations
@app.get("/conversations")
async def get_all_conversations():
    try:
        response = (
            supabase_client.table("conversations")
            .select("*")
            .order("created_at", desc=True)
            .execute()
        )
        return response.data
    except Exception as e:
        print(f"Error getting all conversations: {e}")
        raise HTTPException(status_code=500, detail="Failed to get all conversations.")


# retrieve messages from specific conversation
@app.get("/conversations/{conversation_id}/messages")
async def get_conversation_messages(conversation_id: str):
    try:
        response = (
            supabase_client.table("messages")
            .select("*")
            .eq("conversation_id", conversation_id)
            .order("created_at")
            .execute()
        )
        return response.data
    except Exception as e:
        print(f"Error fetching messages: {e}")
        raise HTTPException(status_code=500, detail="Failed to fetch messages")


# endpoint for uploading a file
@app.post("/conversations/{conversation_id}/uploadfile")
async def process_file(file: UploadFile, conversation_id):

    file_location = await upload_file(file)
    print(f"Saved to: {file_location}")

    extract_text_from_file(file_location, conversation_id)
    print(f"Extraction complete!")

    return {"status": "success", "filename": file.filename}


# uploading a file
async def upload_file(file: UploadFile):
    random_uuid = uuid.uuid4()
    new_filename = f"{random_uuid}_{file.filename}"
    filename = os.path.basename(new_filename)
    file_location = Path(UPLOAD_DIR) / filename

    try:
        async with aiofiles.open(file_location, "wb") as buffer:
            contents = await file.read()
            await buffer.write(contents)
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"There was an error uploading the file: {e}"
        )
    finally:
        await file.close()

    return file_location


# extracting and processing information from file
def extract_text_from_file(file_location: Path, conversation_id):
    ext = file_location.suffix.lower()
    if ext == ".pdf":
        process_pdf(file_location, conversation_id)
    elif ext == ".png" or ext == ".jpg" or ext == ".jpeg":
        generated_text_from_image = process_image_with_text(file_location)

        if len(generated_text_from_image.split()) > 3:
            store_to_both_collections_file(
                conversation_id, generated_text_from_image, file_location
            )
        else:
            generated_caption_of_image = process_image_photo(file_location)
            store_to_both_collections_file(
                conversation_id, generated_caption_of_image, file_location
            )


def process_pdf(file_location, conversation_id):
    """
    The purpose of this function is extract the text from a pdf.

    :param file_location: represents file location of file
    :param conversation_id: current conversation id
    """
    loader = PyPDFLoader(file_location)
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800, chunk_overlap=150, add_start_index=True
    )
    splits = text_splitter.split_documents(documents)
    for index, content in enumerate(splits):
        get_conversation_collection(conversation_id).add_texts(
            texts=[content.page_content],
            ids=[f"{conversation_id}_file_{index}_{datetime.now().timestamp()}"],
            metadatas=[
                {
                    "type": "file",
                    "filename": file_location.name,
                    "conversation_id": conversation_id,
                }
            ],
        )
        get_global_collection().add_texts(
            texts=[f"[File: {file_location.name}] {content.page_content}"],
            ids=[f"global_file_{conversation_id}_{index}_{datetime.now().timestamp()}"],
            metadatas=[
                {
                    "type": "file",
                    "filename": file_location.name,
                    "conversation_id": conversation_id,
                }
            ],
        )


def process_image_with_text(file_location):
    """
    The purpose of this function is process an image that has text to turn into text.

    :param file_location: represents the file location of file .
    """
    file_path = str(file_location)
    image = Image.open(file_path)
    image = image.convert("RGB")
    pixel_values = ocr_processor(image, return_tensors="pt").pixel_values
    generated_ids = ocr_model.generate(pixel_values)
    generated_text = ocr_processor.batch_decode(
        generated_ids, skip_special_tokens=True
    )[0]
    return generated_text


def process_image_photo(file_location):
    """
    The purpose of this function is process an image that has does not have a lot of text to caption it
    so that it can be refered to.

    :param file_location: represents location of file.
    """
    file_path = str(file_location)
    image = Image.open(file_path)
    image = image.convert("RGB")
    inputs = caption_processor(image, return_tensors="pt")
    generated_ids = caption_model.generate(**inputs)
    generated_text = caption_processor.batch_decode(
        generated_ids, skip_special_tokens=True
    )[0]
    return generated_text


def store_to_both_collections_file(conversation_id, generated_text, file_location):
    get_conversation_collection(conversation_id).add_texts(
        texts=[f"[File: {file_location.name}] {generated_text}"],
        ids=[f"{conversation_id}_file_{datetime.now().timestamp()}"],
        metadatas=[
            {
                "type": "file",
                "filename": file_location.name,
                "conversation_id": conversation_id,
            }
        ],
    )
    get_global_collection().add_texts(
        texts=[f"[File: {file_location.name}] {generated_text}"],
        ids=[f"global_file_{conversation_id}_file_{datetime.now().timestamp()}"],
        metadatas=[
            {
                "type": "file",
                "filename": file_location.name,
                "conversation_id": conversation_id,
            }
        ],
    )


"""
#create a relevant title if a title has not been created yet 
def generate_title(conversation_id, user_msg, ai_msg):
    try:
        response = supabase_client.table("conversations").select("title").eq('id', conversation_id).execute()
        data = response.data
        if data and data[0]['title'] is None:
             new_title = ai_model.chat.completions.create(
                model="arcee-ai/trinity-large-preview:free",
                messages=[
                    {"role": "system", "content": "Generate a short, informative 3-5 word title for this conversation. Return ONLY the title, nothing else."},
                    {"role": "user", "content": user_msg + " " + ai_msg}
                ],
                stream=False 
            )
             update_response = supabase_client.table("conversations").update({"title": new_title.choices[0].message.content}).eq("id", conversation_id).execute()
    except Exception as e:
        print(f"Exception {e}")
"""
