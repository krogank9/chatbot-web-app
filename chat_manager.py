from dataclasses import dataclass
from typing import List, Dict, Optional
from datetime import datetime
import json
from pathlib import Path
import uuid

@dataclass
class ChatPersona:
    id: str
    name: str
    system_prompt: str
    greeting: str
    description: str
    created_at: str

@dataclass
class ChatMessage:
    role: str  # 'user' or 'assistant'
    content: str
    timestamp: str
    audio_url: Optional[str] = None

@dataclass
class ChatSession:
    id: str
    persona_id: str
    messages: List[ChatMessage]
    created_at: str
    updated_at: str

class ChatManager:
    def __init__(self, data_dir: str = "data"):
        self.data_dir = Path(data_dir)
        self.personas_file = self.data_dir / "personas.json"
        self.chats_dir = self.data_dir / "chats"
        
        # Create directories
        self.data_dir.mkdir(exist_ok=True)
        self.chats_dir.mkdir(exist_ok=True)
        
        # Load or create default personas
        self.personas = self.load_personas()
        if not self.personas:
            self.create_default_personas()
    
    def load_personas(self) -> Dict[str, ChatPersona]:
        if not self.personas_file.exists():
            return {}
            
        with open(self.personas_file, 'r') as f:
            data = json.load(f)
            return {
                id: ChatPersona(**persona_data)
                for id, persona_data in data.items()
            }
    
    def save_personas(self):
        with open(self.personas_file, 'w') as f:
            json.dump({
                id: {
                    'id': persona.id,
                    'name': persona.name,
                    'system_prompt': persona.system_prompt,
                    'greeting': persona.greeting,
                    'description': persona.description,
                    'created_at': persona.created_at
                }
                for id, persona in self.personas.items()
            }, f, indent=2)
    
    def create_default_personas(self):
        defaults = [
            {
                'name': 'Helpful Assistant',
                'system_prompt': 'You are a helpful and friendly AI assistant.',
                'greeting': 'Hello! How can I help you today?',
                'description': 'A general-purpose helpful assistant'
            },
            {
                'name': 'Python Tutor',
                'system_prompt': 'You are an expert Python programming tutor. Explain concepts clearly and provide helpful examples.',
                'greeting': 'Hi! Ready to learn some Python? What would you like to know about?',
                'description': 'Specialized in teaching Python programming'
            },
            {
                'name': 'Creative Writer',
                'system_prompt': 'You are a creative writing assistant with a flair for storytelling and poetry.',
                'greeting': 'Welcome! Let\'s explore the world of creative writing together.',
                'description': 'Helps with creative writing and storytelling'
            }
        ]
        
        for persona in defaults:
            self.create_persona(**persona)
    
    def create_persona(self, name: str, system_prompt: str, greeting: str, description: str) -> ChatPersona:
        persona = ChatPersona(
            id=str(uuid.uuid4()),
            name=name,
            system_prompt=system_prompt,
            greeting=greeting,
            description=description,
            created_at=datetime.now().isoformat()
        )
        self.personas[persona.id] = persona
        self.save_personas()
        return persona
    
    def get_persona(self, persona_id: str) -> Optional[ChatPersona]:
        return self.personas.get(persona_id)
    
    def create_chat(self, persona_id: str) -> ChatSession:
        persona = self.get_persona(persona_id)
        if not persona:
            raise ValueError("Persona not found")
            
        chat = ChatSession(
            id=str(uuid.uuid4()),
            persona_id=persona_id,
            messages=[
                ChatMessage(
                    role='assistant',
                    content=persona.greeting,
                    timestamp=datetime.now().isoformat()
                )
            ],
            created_at=datetime.now().isoformat(),
            updated_at=datetime.now().isoformat()
        )
        
        self.save_chat(chat)
        return chat
    
    def save_chat(self, chat: ChatSession):
        chat_file = self.chats_dir / f"{chat.id}.json"
        with open(chat_file, 'w') as f:
            json.dump({
                'id': chat.id,
                'persona_id': chat.persona_id,
                'messages': [
                    {
                        'role': msg.role,
                        'content': msg.content,
                        'timestamp': msg.timestamp,
                        'audio_url': msg.audio_url
                    }
                    for msg in chat.messages
                ],
                'created_at': chat.created_at,
                'updated_at': chat.updated_at
            }, f, indent=2)
    
    def load_chat(self, chat_id: str) -> Optional[ChatSession]:
        chat_file = self.chats_dir / f"{chat_id}.json"
        if not chat_file.exists():
            return None
            
        with open(chat_file, 'r') as f:
            data = json.load(f)
            return ChatSession(
                id=data['id'],
                persona_id=data['persona_id'],
                messages=[ChatMessage(**msg) for msg in data['messages']],
                created_at=data['created_at'],
                updated_at=data['updated_at']
            )
    
    def list_chats(self) -> List[Dict]:
        chats = []
        for chat_file in self.chats_dir.glob("*.json"):
            with open(chat_file, 'r') as f:
                data = json.load(f)
                persona = self.get_persona(data['persona_id'])
                chats.append({
                    'id': data['id'],
                    'persona_id': data['persona_id'],
                    'persona_name': persona.name if persona else 'Unknown',
                    'created_at': data['created_at'],
                    'updated_at': data['updated_at'],
                    'message_count': len(data['messages'])
                })
        return sorted(chats, key=lambda x: x['updated_at'], reverse=True)
    
    def add_message(self, chat_id: str, role: str, content: str, audio_url: Optional[str] = None) -> ChatSession:
        chat = self.load_chat(chat_id)
        if not chat:
            raise ValueError("Chat not found")
            
        chat.messages.append(ChatMessage(
            role=role,
            content=content.strip(),
            timestamp=datetime.now().isoformat(),
            audio_url=audio_url
        ))
        chat.updated_at = datetime.now().isoformat()
        
        self.save_chat(chat)
        return chat
    
    def update_persona(self, persona_id: str, name: str, system_prompt: str, greeting: str, description: str) -> ChatPersona:
        if persona_id not in self.personas:
            raise ValueError("Persona not found")
        
        persona = ChatPersona(
            id=persona_id,
            name=name,
            system_prompt=system_prompt,
            greeting=greeting,
            description=description,
            created_at=self.personas[persona_id].created_at
        )
        self.personas[persona_id] = persona
        self.save_personas()
        return persona
    
    def delete_chat(self, chat_id: str):
        """Delete a chat and its file."""
        chat_file = self.chats_dir / f"{chat_id}.json"
        if not chat_file.exists():
            raise ValueError("Chat not found")
        chat_file.unlink() 