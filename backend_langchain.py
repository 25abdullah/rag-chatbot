from fastapi import FastAPI, WebSocket, HTTPException, UploadFile
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from supabase import create_client
from openai import OpenAI
from sentence_transformers import SentenceTransformer
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


#initialize app and set up cors 
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
SUPABASE_KEY =  os.getenv("SUPABASE_KEY")
supabase_client = create_client(SUPABASE_URL, SUPABASE_KEY)
OPENROUTER_KEY = os.getenv("OPENROUTER_API_KEY")

ai_model = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key= OPENROUTER_KEY
)

embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")




app.mount("/static", StaticFiles(directory="static"), name="static")


SYSTEM_PROMPT ="""
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




#conversation -> voice 
#Given a converastion id, allows for users to chat with AI with streaming 
@app.websocket("/conversations/{conversation_id}/chat")
async def chat_websocket(websocket: WebSocket, conversation_id: str):
    await websocket.accept() #get connection 
    try:
        while True:
            data = await websocket.receive_text() 
            #retrieve top 5 most similar from same conversation 
            same_conversation_retrieval = get_conversation_collection(conversation_id).similarity_search(
                data,
                k = 5
            )
            #retrieve top 3 most similar from across all conversations 
            global_retrieval = get_global_collection().similarity_search (
                data,
                k = 3
            )
            #build up the context 
            memory_text = "" 
            if same_conversation_retrieval:
                for res in same_conversation_retrieval:
                    memory_text += res.page_content + " "
            if global_retrieval:
                for res in global_retrieval:
                    memory_text += res.page_content + " "

            if not memory_text:
                memory_text = "No previous conversations yet. This is a fresh start!"
            
            #provide to prompt 
            full_system_prompt = SYSTEM_PROMPT.format(memory_text=memory_text)
            
            #make AI response 
            response = ai_model.chat.completions.create(
                model="arcee-ai/trinity-large-preview:free",
                messages=[
                    {"role": "system", "content": full_system_prompt},
                    {"role": "user", "content": data}
                ],
                stream=True
            )
            #build AI response 
            build_response = ""
            for chunk in response:
                token = chunk.choices[0].delta.content
                if token:
                    build_response += token
                    await websocket.send_text(token)
            
            #create title 
            generate_title(conversation_id, data, build_response)
            user_data_to_insert = {
                "conversation_id": conversation_id,
                "role": "user",
                "content": data
            }
            ai_data_to_insert = {
                "conversation_id": conversation_id,
                "role": "assistant",
                "content": build_response
            }


            #insert user + ai data into supabase 
            supabase_client.table("messages").insert([
                user_data_to_insert, 
                ai_data_to_insert
            ]).execute()


            #embed user message 
            get_conversation_collection(conversation_id).add_texts(
                texts=[data],
                ids=[f"{conversation_id}_user_{datetime.now().timestamp()}"])
            #embed ai message 
            get_conversation_collection(conversation_id).add_texts(
                texts=[build_response],
                ids=[f"{conversation_id}_ai_{datetime.now().timestamp()}"])    
            
            #chunk c
            chunk_to_global(conversation_id)   
    except Exception as e:
        print(f"Exception {e}")

#return full conversation given an id for chroma side (not supabase)
def get_conversation_collection(conversation_id):
    return Chroma(
        persist_directory="./chroma_db",
        embedding_function=embeddings,
        collection_name=f"conv_{conversation_id}"
    )

#return all conversations 
def get_global_collection():
    return Chroma(
        persist_directory="./chroma_db",
        embedding_function=embeddings,
        collection_name="user_123_global"
    )

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



#chunk every 10 messages into global collection for richer cross-conversation context
#instead of storing individual messages, we combine 10 messages into one embedding
#this provides more complete context when retrieving from global memory
def chunk_to_global(conversation_id):
     try:
        print(f"Checking if should chunk for conversation: {conversation_id}")
        response = (
            supabase_client.table("messages")
            .select("*", count="exact")
            .eq("conversation_id", conversation_id)
            .execute()
        )
        print(f"🔍 Message count: {response.count}")
        print(f"🔍 Count % 10 = {response.count % 10}")
        
        if (response.count % 10 == 0):
            print(f"Chunking to global! Message count: {response.count}")
            get_last_ten = (
                supabase_client.table("messages")
                .select("*")
                .eq("conversation_id", conversation_id)
                .order("created_at", desc=True) 
                .limit(10)
                .execute()
            )
            group_messages = ""
            rows = get_last_ten.data
            for row in rows:
                group_messages += row['content']
            get_global_collection().add_texts(
                texts=[group_messages], 
                ids=[f"{conversation_id}_{datetime.now().timestamp()}"]
            )
            print("Successfully chunked to global!")
     except Exception as e:
         print(f"Exception in chunk_to_global: {e}")




#create a conversation
@app.post("/conversations")
async def create_conversation():
    try:
        response = supabase_client.table("conversations").insert({}).execute()
        created_conversation = response.data[0]
        return created_conversation
    except Exception as e:
        print(f"Exception {e}")

#delete conversation and corresponding data 
@app.delete("/conversations/{conversation_id}")
async def delete_conversation(conversation_id: str):
    try: 
        supabase_client.from_("messages").delete().eq("conversation_id", conversation_id).execute()
        supabase_client.table("conversations").delete().eq("id", conversation_id).execute()
        return {"status": "success", "message": f"Deleted conversation {conversation_id}"}
    except Exception as e:
        print(f"Error deleting conversation id: {conversation_id}")
        raise HTTPException(status_code = 500, detail = "Failed to delete conversation.")

#return all conversations
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


#retrieve messages from specific conversation 
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



#endpoint for uploading a file 
@app.post("/conversations/{conversation_id}/uploadfile")
async def process_file(file: UploadFile, conversation_id):
   
    file_location = await upload_file(file)
    print(f"Saved to: {file_location}")
    
    extract_text_from_file(file_location, conversation_id)
    print(f"Extraction complete!")
    
    return {"status": "success", "filename": file.filename}

UPLOAD_DIR = "uploaded_files"
os.makedirs(UPLOAD_DIR, exist_ok=True)


#uploading a file 
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
        raise HTTPException( status_code=500,
            detail=f"There was an error uploading the file: {e}"
        )
    finally:
        await file.close()

    return file_location

#extracting and processing information from file 
def extract_text_from_file(file_location: Path, conversation_id):
 loader = PyPDFLoader(file_location)
 documents = loader.load()
 text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=800, 
    chunk_overlap=150, 
    add_start_index=True)
 splits = text_splitter.split_documents(documents)
 for index, content in enumerate(splits): 
    get_conversation_collection(conversation_id).add_texts(
                texts=[content.page_content],
                ids=[f"{conversation_id}_file_{index}_{datetime.now().timestamp()}"])
    get_global_collection().add_texts(
            texts=[f"[File: {file_location.name}] {content.page_content}"],
            ids=[f"global_file_{conversation_id}_{index}_{datetime.now().timestamp()}"]
        )


#testing endpoint 
@app.post("/test-upload")
async def test_upload(file: UploadFile):
    return {"filename": file.filename, "size": file.size}