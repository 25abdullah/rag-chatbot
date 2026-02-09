
//constants 
const sidebar = document.querySelector('.sidebar');
const messageArea = document.getElementById('chat-box');
const sendButton = document.getElementById('button');
const newChatButton = document.getElementById('new-chat-btn');
const userMessage = document.getElementById('messageInput');
const chatList = document.getElementById('chat-list');
const uploadBtn = document.getElementById('uploadBtn');
const fileInput = document.getElementById('fileInput');

let ws = null;
let currentConversationId = null 
let currentAIMessage = null;


//initially disable sendbutton until chat button is created 
sendButton.disabled = true;


//load in conversations on the side bar 
async function loadConversations() {
    const response = await fetch('http://localhost:8000/conversations');
    const data = await response.json();
    chatList.textContent = '';
    
    for (let i = 0; i < data.length; i++) {
        // Create container with flex layout
        const containerDiv = document.createElement('div');
        containerDiv.style.display = 'flex';
        containerDiv.style.alignItems = 'center';
        containerDiv.style.gap = '5px';
        containerDiv.style.marginBottom = '5px';
        
        // Create conversation button
        const newElement = document.createElement('button');
        newElement.textContent = data[i].title || 'Chat';
        newElement.style.flex = '1';
        newElement.style.textAlign = 'left';
        newElement.style.padding = '15px';
        newElement.style.border = 'none';
        newElement.style.background = 'white';
        newElement.style.cursor = 'pointer';
        newElement.addEventListener('click', function() {
            selectConversation(data[i].id);  
        });
        
        // Create delete button
        const deleteBlock = document.createElement('button');
        deleteBlock.textContent = '🗑️';
        deleteBlock.style.width = '40px';
        deleteBlock.style.height = '40px';
        deleteBlock.style.padding = '0';
        deleteBlock.style.border = 'none';
        deleteBlock.style.background = '#ff4444';
        deleteBlock.style.color = 'white';
        deleteBlock.style.cursor = 'pointer';
        deleteBlock.style.borderRadius = '4px';
        deleteBlock.addEventListener('click', function(e) {
            e.stopPropagation();
            if (confirm('Delete this conversation?')) {
                deleteConversation(data[i].id);
            }
        });
        
        // Add both to container
        containerDiv.appendChild(newElement);
        containerDiv.appendChild(deleteBlock);
        chatList.appendChild(containerDiv); 
    }
}

// Function to handle conversation click
async function selectConversation(conversationId) {
    currentConversationId = conversationId
    const allConvs = document.querySelectorAll('#chat-list > div');
    allConvs.forEach(div => {
        div.style.backgroundColor = '';  
    });
    
    event.currentTarget.parentElement.style.backgroundColor = '#d0e8ff';  
    
    //close current 
    if (ws) {
        ws.close();
    }
    
    //clear messages 
    messageArea.textContent = '';
    
    //pause everything to load in messages 
    await loadMessages(currentConversationId);
    
    //then connect 
    connectWebSocket(currentConversationId);
}



//listener for uploading a file 
uploadBtn.addEventListener('click', function() {
    fileInput.click();
});

//when file is clicked 
fileInput.addEventListener('change', async function() {
    if (fileInput.files.length > 0) {
        const file = fileInput.files[0];
        
        const formData = new FormData();
        formData.append('file', file);
        
        const response = await fetch(`http://localhost:8000/conversations/${currentConversationId}/uploadfile`, {
            method: 'POST',
            body: formData
        });
        
        const result = await response.json();
        alert(`File uploaded: ${result.filename}`);
        
        fileInput.value = ''; 
    }
});

// Function to fetch and display messages
async function loadMessages(conversationId) {
    currentConversationId = conversationId
    const response = await fetch(`http://localhost:8000/conversations/${conversationId}/messages`);
    console.log("Response status:", response.status);  
    const messages = await response.json();
    console.log("Messages:", messages);  
    messageArea.textContent = '';
    for (let i = 0; i < messages.length; i++) {
        displayMessage(messages[i].role, messages[i].content)

}
}

//connect the websocket and receive ai message 
function connectWebSocket(conversationId) {
     if (ws) { //close if we have current opening 
        ws.close();  
    }
    ws = new WebSocket(`ws://localhost:8000/conversations/${conversationId}/chat`);
    sendButton.disabled = false;
    
    ws.onmessage = (event) => {
        // If no current AI message, create one
        if (!currentAIMessage) {
            currentAIMessage = document.createElement('li');
            currentAIMessage.style.textAlign = 'left';
            currentAIMessage.style.background = '#f0f0f0';
            messageArea.appendChild(currentAIMessage);
        }
        
        // Append new token to existing message
        currentAIMessage.textContent += event.data;
        messageArea.scrollTop = messageArea.scrollHeight;
    };
}

//send user message 
function sendMessage() {
    currentAIMessage = null;  // Reset for new AI  response
    displayMessage('user', userMessage.value);
    ws.send(userMessage.value);
    userMessage.value = '';
}

// Function to display a message in chat
function displayMessage(role, content) {
    
    const li = document.createElement('li');
    if (role === 'user') {
        li.style.textAlign = 'right';
        li.style.background = '#007bff';
        li.style.color = 'white';
    } else {
        li.style.textAlign = 'left';
        li.style.background = '#f0f0f0';
    }
    li.textContent = content
     messageArea.appendChild(li);
    messageArea.scrollTop = messageArea.scrollHeight;
}


//create a new chat 
async function createNewConversation() {
    const response = await fetch('http://localhost:8000/conversations', {
        method: 'POST'
    });
    const newConvo = await response.json();
    
    selectConversation(newConvo.id);
    loadConversations();
}

//delete a conversation 
async function deleteConversation(conversationId) {
    await fetch(`http://localhost:8000/conversations/${conversationId}`, {
        method: 'DELETE'
    });
    loadConversations();
       // If we deleted the current conversation, clear the chat
    if (conversationId === currentConversationId) {
        currentConversationId = null;
        messageArea.textContent = '';  // Clear messages
        if (ws) {
            ws.close();  // Close WebSocket
        }
        sendButton.disabled = true;  // Disable send
    }
    
    loadConversations();  // Refresh sidebar
}

// Run on page load
window.onload = function() {
    loadConversations();
};

sendButton.addEventListener('click', sendMessage);
newChatButton.addEventListener('click', createNewConversation)

selectConversation
