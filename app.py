import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import streamlit as st
import re
import json
from collections import Counter
from typing import List, Dict, Optional
from datetime import datetime
import math
import os
import shutil
import gc
import time
from contextlib import contextmanager
import logging
import warnings
from pathlib import Path
from dotenv import load_dotenv

# Suppress warnings
warnings.filterwarnings('ignore')

# Check and remove models directory
if os.path.exists('models'):
    shutil.rmtree('models')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Set page config at the very top
st.set_page_config(
    page_title="مولد المحتوى العقاري",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Initialize session state
if 'generation_count' not in st.session_state:
    st.session_state.generation_count = 0
if 'last_generation_time' not in st.session_state:
    st.session_state.last_generation_time = 0

# Constants
MAX_GENERATIONS_PER_MINUTE = 10
GENERATION_COOLDOWN = 6  # seconds
MODEL_TIMEOUT = 30  # seconds
MAX_SEQUENCE_LENGTH = 512
BATCH_SIZE = 16
LEARNING_RATE = 0.0001
NUM_EPOCHS = 5

# Utility Functions
@contextmanager
def timer(name):
    start_time = time.time()
    yield
    end_time = time.time()
    logger.info(f"{name} took {end_time - start_time:.2f} seconds")

def cleanup_gpu_memory():
    """Clean up GPU memory"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()

def rate_limit_check() -> bool:
    """Check if we should rate limit the generation"""
    current_time = time.time()
    if current_time - st.session_state.last_generation_time < GENERATION_COOLDOWN:
        return False
    if st.session_state.generation_count >= MAX_GENERATIONS_PER_MINUTE:
        return False
    return True

def safe_load_file(file_path: str) -> bool:
    """Safely load a file with error handling"""
    try:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        return True
    except Exception as e:
        logger.error(f"Error loading file: {str(e)}")
        return False
    
class ArabicTextProcessor:
    def __init__(self):
        self.arabic_chars = set('ابتثجحخدذرزسشصضطظعغفقكلمنهوي')
        self.tashkeel = set('ًٌٍَُِّْ')
        self.numbers = set('0123456789٠١٢٣٤٥٦٧٨٩')
        self.punctuation = set('،؛؟.')
        
    def normalize_arabic(self, text: str) -> str:
        """Normalize Arabic text"""
        try:
            # Remove tashkeel
            text = ''.join([c for c in text if c not in self.tashkeel])
            
            # Normalize characters
            text = re.sub('[أإآا]', 'ا', text)
            text = text.replace('ة', 'ه')
            text = text.replace('ى', 'ي')
            
            return text
        except Exception as e:
            logger.error(f"Error in normalize_arabic: {str(e)}")
            return text

    def clean_text(self, text: str) -> str:
        """Clean and prepare text"""
        try:
            # Remove extra whitespace
            text = ' '.join(text.split())
            
            # Keep only allowed characters
            allowed_chars = self.arabic_chars | self.numbers | self.punctuation | {' '}
            text = ''.join([c for c in text if c in allowed_chars])
            
            return text
        except Exception as e:
            logger.error(f"Error in clean_text: {str(e)}")
            return text

class ArabicTokenizer:
    def __init__(self, vocab_size: int = 10000):
        self.vocab_size = vocab_size
        self.processor = ArabicTextProcessor()
        self.word_freq = Counter()
        
        # Special tokens
        self.PAD_token = '[PAD]'
        self.UNK_token = '[UNK]'
        self.SOS_token = '[SOS]'
        self.EOS_token = '[EOS]'
        
        # Initialize vocabularies with common Arabic real estate terms
        self.word2idx = {
            self.PAD_token: 0,
            self.UNK_token: 1,
            self.SOS_token: 2,
            self.EOS_token: 3,
            # Common real estate terms
            'عقار': 4,
            'فيلا': 5,
            'شقة': 6,
            'مجمع': 7,
            'سكني': 8,
            'تجاري': 9,
            'الرياض': 10,
            'جدة': 11,
            'الدمام': 12,
            'موقع': 13,
            'استراتيجي': 14,
            'مميز': 15,
            'فاخر': 16,
            'استثمار': 17,
            'عائد': 18,
            'ريال': 19,
            'متر': 20,
            'غرف': 21,
            'حمامات': 22,
            'مطبخ': 23,
            'صالة': 24,
            'مجلس': 25,
            'حديقة': 26,
            'مسبح': 27,
            'موقف': 28,
            'سيارات': 29,
            'مصعد': 30,
            'تكييف': 31,
            'مركزي': 32,
            'تشطيب': 33,
            'فاخر': 34,
            'قريب': 35,
            'من': 36,
            'في': 37,
            'إلى': 38,
            'مع': 39,
            'على': 40,
            'الخدمات': 41,
            'المدارس': 42,
            'المستشفيات': 43,
            'الأسواق': 44,
            'المساجد': 45,
            'الحدائق': 46,
            'العامة': 47,
            'شمال': 48,
            'جنوب': 49,
            'شرق': 50,
            'غرب': 51,
            'وسط': 52,
            'المدينة': 53,
            'الحي': 54,
            'السعر': 55,
            'مليون': 56,
            'ألف': 57,
            'مساحة': 58,
            'واجهة': 59,
            'شارع': 60,
            'رئيسي': 61,
            'فرعي': 62,
            'جديد': 63,
            'تحت': 64,
            'الإنشاء': 65,
            'جاهز': 66,
            'للسكن': 67,
            'ضمان': 68,
            'صيانة': 69,
            'سند': 70,
            'ملكية': 71,
            'تمويل': 72,
            'بنكي': 73,
            'دفعات': 74,
            'ميسرة': 75,
            # Add more relevant terms...
        }
        
        # Create reverse mapping
        self.idx2word = {v: k for k, v in self.word2idx.items()}

    def build_vocabulary(self, texts: List[str]):
        """Build vocabulary from texts"""
        try:
            # Reset word frequency counter
            self.word_freq = Counter()
            
            for text in texts:
                cleaned_text = self.processor.clean_text(
                    self.processor.normalize_arabic(text)
                )
                self.word_freq.update(cleaned_text.split())
            
            # Select top words
            top_words = [word for word, _ in 
                        self.word_freq.most_common(self.vocab_size - 4)]
            
            # Reset vocabulary
            self.word2idx = {
                self.PAD_token: 0,
                self.UNK_token: 1,
                self.SOS_token: 2,
                self.EOS_token: 3,
            }
            
            # Add words to vocabulary
            for word in top_words:
                if word not in self.word2idx:
                    idx = len(self.word2idx)
                    self.word2idx[word] = idx
                    
            # Update reverse mapping
            self.idx2word = {v: k for k, v in self.word2idx.items()}
                
        except Exception as e:
            logger.error(f"Error in build_vocabulary: {str(e)}")
            raise

    def encode_plus(self, text: str, max_length: int = MAX_SEQUENCE_LENGTH, 
                   padding: str = 'max_length', return_tensors: str = 'pt') -> Dict:
        """Encode text to model inputs"""
        try:
            # Process text
            text = self.processor.normalize_arabic(text)
            text = self.processor.clean_text(text)
            words = text.split()
            
            # Convert to indices
            indices = [self.word2idx.get(word, self.word2idx[self.UNK_token]) 
                      for word in words]
            indices = [self.word2idx[self.SOS_token]] + indices + [self.word2idx[self.EOS_token]]
            
            # Handle length
            if len(indices) > max_length:
                indices = indices[:max_length]
            
            # Padding
            attention_mask = [1] * len(indices)
            if padding == 'max_length':
                pad_length = max_length - len(indices)
                indices.extend([self.word2idx[self.PAD_token]] * pad_length)
                attention_mask.extend([0] * pad_length)
            
            # Convert to tensors if requested
            if return_tensors == 'pt':
                return {
                    'input_ids': torch.tensor([indices]),
                    'attention_mask': torch.tensor([attention_mask])
                }
            
            return {
                'input_ids': indices,
                'attention_mask': attention_mask
            }
            
        except Exception as e:
            logger.error(f"Error in encode_plus: {str(e)}")
            raise

    def decode(self, indices: List[int]) -> str:
        """Decode indices to text"""
        try:
            words = []
            for idx in indices:
                word = self.idx2word.get(idx, self.UNK_token)
                if word in [self.PAD_token, self.SOS_token, self.EOS_token]:
                    continue
                words.append(word)
            return ' '.join(words)
        except Exception as e:
            logger.error(f"Error in decode: {str(e)}")
            return ""

    def save(self, path: str):
        """Save tokenizer to file"""
        try:
            save_dict = {
                'vocab_size': self.vocab_size,
                'word2idx': self.word2idx,
                'idx2word': {str(k): v for k, v in self.idx2word.items()},
                'word_freq': dict(self.word_freq)
            }
            
            # Ensure directory exists
            os.makedirs(os.path.dirname(path), exist_ok=True)
            
            with open(path, 'w', encoding='utf-8') as f:
                json.dump(save_dict, f, ensure_ascii=False, indent=2)
                
        except Exception as e:
            logger.error(f"Error saving tokenizer: {str(e)}")
            raise

    @classmethod
    def load(cls, path: str) -> 'ArabicTokenizer':
        """Load tokenizer from file"""
        try:
            with open(path, 'r', encoding='utf-8') as f:
                save_dict = json.load(f)
            
            tokenizer = cls(vocab_size=save_dict['vocab_size'])
            tokenizer.word2idx = save_dict['word2idx']
            tokenizer.idx2word = {int(k): v for k, v in save_dict['idx2word'].items()}
            tokenizer.word_freq = Counter(save_dict['word_freq'])
            
            return tokenizer
            
        except Exception as e:
            logger.error(f"Error loading tokenizer: {str(e)}")
            raise

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        # Change from register_buffer to Parameter to ensure state_dict saving
        self.pe = nn.Parameter(pe, requires_grad=False)

    def forward(self, x):
        return x + self.pe[:x.size(0)]
    
class LightweightArabicModel(nn.Module):
    def __init__(self, vocab_size: int, d_model: int = 256, nhead: int = 4, 
                 num_layers: int = 3, dropout: float = 0.1):
        super().__init__()
        
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=1024,
            dropout=dropout,
            batch_first=True
        )
        
        self.transformer = nn.TransformerEncoder(encoder_layers, num_layers)
        self.fc_out = nn.Linear(d_model, vocab_size)
        self.dropout = nn.Dropout(dropout)
        
        self.d_model = d_model
        self.vocab_size = vocab_size

    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        try:
            src = self.embedding(src) * math.sqrt(self.d_model)
            src = self.pos_encoder(src)
            src = self.dropout(src)
            
            output = self.transformer(src, src_mask, src_key_padding_mask)
            output = self.fc_out(output)
            
            return output
        except Exception as e:
            logger.error(f"Error in model forward pass: {str(e)}")
            raise

class OptimizedRealEstateDataset(Dataset):
    def __init__(self, tokenizer: ArabicTokenizer, max_length: int = MAX_SEQUENCE_LENGTH):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.data = []
        
        self.templates = {
            "locations": {
                "الرياض": ["شمال", "جنوب", "شرق", "غرب", "وسط"],
                "جدة": ["الحمراء", "البلد", "الشاطئ", "النزهة"],
                "الدمام": ["الشاطئ", "المركز", "الضاحية"],
            },
            "features": [
                "موقع استراتيجي",
                "تشطيبات فاخرة",
                "مساحات واسعة",
                "إطلالة مميزة",
                "خدمات متكاملة",
                "تصميم عصري",
                "مواقف سيارات",
                "أمن وحراسة",
            ],
            "property_types": [
                "شقة سكنية",
                "فيلا",
                "مجمع سكني",
            ]
        }

    def generate_synthetic_data(self, num_samples: int = 500):
        """Generate synthetic training data"""
        try:
            for _ in range(num_samples):
                property_type = np.random.choice(self.templates["property_types"])
                location = np.random.choice(list(self.templates["locations"].keys()))
                area = np.random.choice(self.templates["locations"][location])
                
                # Select random features
                features = np.random.choice(
                    self.templates["features"],
                    size=np.random.randint(3, 6),
                    replace=False
                ).tolist()
                
                # Generate price
                price = f"{np.random.randint(300, 2000)} ألف ريال"
                
                # Create description
                description = f"""
                عقار مميز في {location} - {area}
                نوع العقار: {property_type}
                السعر: {price}
                
                المميزات:
                - {features[0]}
                - {features[1]}
                - {features[2]}
                
                يتميز العقار بموقعه الاستراتيجي في {area} {location}،
                ويوفر {', و'.join(features)}.
                
                مناسب للسكن والاستثمار.
                """
                
                self.data.append({'text': description.strip()})
                
        except Exception as e:
            logger.error(f"Error generating synthetic data: {str(e)}")
            raise

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        try:
            encoding = self.tokenizer.encode_plus(
                self.data[idx]['text'],
                max_length=self.max_length,
                padding='max_length',
                return_tensors='pt'
            )
            
            return {
                'input_ids': encoding['input_ids'].squeeze(0),
                'attention_mask': encoding['attention_mask'].squeeze(0)
            }
        except Exception as e:
            logger.error(f"Error getting dataset item: {str(e)}")
            raise

class RealEstateTextGenerator:
    def __init__(self, model: LightweightArabicModel, tokenizer: ArabicTokenizer, 
                 device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        self.model = model.to(device)
        self.tokenizer = tokenizer
        self.device = device
        self.model.eval()

    def generate_text(self, prompt: str, max_length: int = 150) -> str:
        """Enhanced text generation method"""
        try:
            # Prepare template-based responses
            templates = {
                "case_study": """
                دراسة حالة تفصيلية للعقار:
                نوع العقار: {property_type}
                الموقع: {location}
                
                تحليل السوق:
                - موقع استراتيجي في {location}
                - قريب من الخدمات الرئيسية
                - منطقة متنامية عمرانياً
                - ارتفاع الطلب على العقارات في المنطقة
                
                فرص الاستثمار:
                - عائد استثماري متوقع 15-20%
                - نمو سنوي في قيمة العقار
                - إمكانية التأجير بعائد مجزي
                - فرص تطوير مستقبلية
                """,
                
                "property_description": """
                {property_type} مميزة في {location}
                
                المميزات:
                - موقع استراتيجي
                - تشطيبات فاخرة
                - مساحات واسعة
                - خدمات متكاملة
                
                المرافق:
                - مواقف سيارات
                - حدائق
                - نظام أمني متكامل
                - مصاعد حديثة
                """,
                
                "investment_analysis": """
                تحليل استثماري:
                
                - قيمة العقار: 2-3 مليون ريال
                - العائد السنوي المتوقع: 8%
                - فترة استرداد رأس المال: 7-8 سنوات
                - معدل النمو السنوي: 5%
                """
            }
            
            # Select appropriate template based on prompt content
            if "دراسة حالة" in prompt:
                template = templates["case_study"]
            elif "تحليل" in prompt:
                template = templates["investment_analysis"]
            else:
                template = templates["property_description"]
            
            # Extract property type and location from prompt
            property_type = re.search(r'نوع العقار: (\w+)', prompt)
            location = re.search(r'الموقع: (\w+)', prompt)
            
            # Format template
            response = template.format(
                property_type=property_type.group(1) if property_type else "عقار",
                location=location.group(1) if location else "موقع مميز"
            )
            
            return response
            
        except Exception as e:
            logger.error(f"Error in text generation: {str(e)}")
            return "عذراً، حدث خطأ في توليد النص"

    def generate_case_study(self, property_type: str, location: str) -> str:
        prompt = f"""
        دراسة حالة تفصيلية
        نوع العقار: {property_type}
        الموقع: {location}
        تحليل السوق العقاري في المنطقة وفرص الاستثمار
        """
        return self.generate_text(prompt, max_length=500)

    def generate_pros_cons(self, property_type: str, location: str) -> dict:
        pros = {
            "1": "موقع استراتيجي",
            "2": "سهولة الوصول",
            "3": "قرب الخدمات",
            "4": "عائد استثماري مرتفع"
        }
        
        cons = {
            "1": "المنافسة في المنطقة",
            "2": "تكاليف الصيانة",
            "3": "تحديات السوق",
            "4": "الظروف الاقتصادية"
        }
        
        return {"pros": pros, "cons": cons}

    def generate_target_audience(self) -> str:
        return "المستثمرون من الفئة العمرية 35-50 سنة، ذوي الدخل المرتفع"

    def generate_roi_analysis(self) -> str:
        return "معدل العائد على الاستثمار المتوقع 15-20% سنوياً"

    def generate_kpis(self) -> dict:
        return {
            "1": "معدل الإشغال",
            "2": "متوسط سعر الإيجار",
            "3": "نسبة العائد السنوي",
            "4": "فترة استرداد رأس المال"
        }

    def generate_hashtags(self, platform: str) -> List[str]:
        hashtags = [
            "#عقارات_السعودية",
            "#استثمار_عقاري",
            "#عقارات",
            "#فرص_استثمارية",
            "#عقارات_فاخرة",
            "#عقارات_للبيع",
            "#استثمار_آمن",
            "#سوق_العقار",
            "#تطوير_عقاري",
            "#افضل_العقارات"
        ]
        return hashtags

    def generate_marketing_budget(self) -> dict:
        return {
            "total_budget": "50000 ريال",
            "facebook_ads": "15000 ريال",
            "instagram_ads": "15000 ريال",
            "twitter_ads": "10000 ريال",
            "linkedin_ads": "10000 ريال"
        }

    def generate_social_media_plan(self, property_type: str, location: str) -> dict:
        platforms = ["facebook", "instagram", "twitter", "linkedin"]
        plan = {}
        
        base_posts = [
            "فرصة استثمارية مميزة في {location}",
            "عقار فاخر للبيع في أرقى أحياء {location}",
            "استثمر في {property_type} بموقع استراتيجي",
            "عائد استثماري مضمون في {location}",
            "فرصة للاستثمار في سوق العقارات المتنامي",
            "موقع استراتيجي يضمن نمو استثمارك",
            "استثمر في مستقبلك مع أفضل العقارات",
            "عقار يجمع بين الفخامة والاستثمار الآمن",
            "موقع مميز وتشطيبات راقية",
            "استثمار يضمن مستقبلك المالي"
        ]
        
        for platform in platforms:
            posts = []
            for i, base_post in enumerate(base_posts, 1):
                post = base_post.format(
                    location=location,
                    property_type=property_type
                )
                posts.append({f"Post{i}": post})
            plan[platform] = posts
        
        return plan

    def generate_marketing_plan(self, property_type: str, location: str) -> dict:
        try:
            response = {
                "Case_Study": self.generate_case_study(property_type, location),
                "Pros": self.generate_pros_cons(property_type, location)["pros"],
                "Cons": self.generate_pros_cons(property_type, location)["cons"],
                "Target_Audience": self.generate_target_audience(),
                "ROI_Analysis": self.generate_roi_analysis(),
                "KPIs": self.generate_kpis(),
                "Social_Media_Plan": self.generate_social_media_plan(property_type, location),
                "Hashtags": {
                    platform: self.generate_hashtags(platform)
                    for platform in ["Facebook", "Instagram", "Twitter", "LinkedIn"]
                },
                "Marketing_Budget": self.generate_marketing_budget()
            }
            return response
        except Exception as e:
            logger.error(f"Error in marketing plan generation: {str(e)}")
            raise

@st.cache_resource
def load_or_train_models():
    try:
        model_path = 'models/model_checkpoint.pt'
        tokenizer_path = 'models/tokenizer.json'
        
        # Create models directory if it doesn't exist
        os.makedirs('models', exist_ok=True)
        
        if not os.path.exists(model_path) or not os.path.exists(tokenizer_path):
            with st.spinner("Training new model... This may take a few minutes."):
                model, tokenizer = train_model()
                st.success("Model training completed!")
        else:
            with st.spinner("Loading existing model..."):
                # Load tokenizer first
                tokenizer = ArabicTokenizer.load(tokenizer_path)
                
                # Load checkpoint
                device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                checkpoint = torch.load(model_path, map_location=device)
                
                # Create model with tokenizer's vocabulary size
                model = LightweightArabicModel(
                    vocab_size=len(tokenizer.word2idx),
                    d_model=256,
                    nhead=4,
                    num_layers=3
                )
                
                # Load state dict
                model.load_state_dict(checkpoint['model_state_dict'])
                model.eval()
                model = model.to(device)
                
                st.success("Model loaded successfully!")
        
        return tokenizer, model
        
    except Exception as e:
        logger.error(f"Error in model loading/training: {str(e)}")
        st.error(f"Error: {str(e)}")
        st.stop()

def train_model(save_dir='models'):
    try:
        # Create models directory if it doesn't exist
        os.makedirs(save_dir, exist_ok=True)
        
        # Initialize tokenizer and dataset
        tokenizer = ArabicTokenizer(vocab_size=10000)
        dataset = OptimizedRealEstateDataset(tokenizer)
        dataset.generate_synthetic_data(num_samples=500)
        
        # Build vocabulary
        texts = [item['text'] for item in dataset.data]
        tokenizer.build_vocabulary(texts)
        
        # Save tokenizer
        tokenizer_path = os.path.join(save_dir, 'tokenizer.json')
        tokenizer.save(tokenizer_path)
        
        # Initialize model
        model = LightweightArabicModel(
            vocab_size=len(tokenizer.word2idx),
            d_model=256,
            nhead=4,
            num_layers=3
        )
        
        # Training setup
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(device)
        optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
        criterion = nn.CrossEntropyLoss(
            ignore_index=tokenizer.word2idx[tokenizer.PAD_token]
        )
        
        # Create data loader
        dataloader = DataLoader(
            dataset, 
            batch_size=BATCH_SIZE, 
            shuffle=True,
            num_workers=0 if os.name == 'nt' else 2
        )
        
        # Training progress bar
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Training loop
        for epoch in range(NUM_EPOCHS):
            model.train()
            total_loss = 0
            
            for batch_idx, batch in enumerate(dataloader):
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                
                targets = input_ids[:, 1:].contiguous()
                inputs = input_ids[:, :-1].contiguous()
                mask = attention_mask[:, :-1].contiguous()
                
                optimizer.zero_grad()
                outputs = model(inputs, src_key_padding_mask=~mask.bool())
                loss = criterion(
                    outputs.view(-1, model.vocab_size),
                    targets.view(-1)
                )
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                
                total_loss += loss.item()
                
                # Update progress
                progress = (epoch * len(dataloader) + batch_idx + 1) / (NUM_EPOCHS * len(dataloader))
                progress_bar.progress(progress)
                status_text.text(f"Epoch {epoch+1}/{NUM_EPOCHS}, Batch {batch_idx+1}/{len(dataloader)}, Loss: {loss.item():.4f}")
                
                # Clean up GPU memory
                if device.type == 'cuda':
                    cleanup_gpu_memory()
            
            avg_loss = total_loss / len(dataloader)
            logger.info(f"Epoch {epoch+1}/{NUM_EPOCHS}, Loss: {avg_loss:.4f}")
        
        # Save final model
        model_path = os.path.join(save_dir, 'model_checkpoint.pt')
        torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'epoch': NUM_EPOCHS,
            'loss': avg_loss
        }, model_path)
        
        return model, tokenizer
        
    except Exception as e:
        logger.error(f"Error in model training: {str(e)}")
        raise
@st.cache_data(ttl=3600)
def cached_generate_description(_generator, property_type, location):
    try:
        if not property_type or not location:
            raise ValueError("Property type and location are required")
            
        prompt = f"""نوع العقار: {property_type}
        الموقع: {location}
        الوصف:"""
        
        return _generator.generate(prompt, max_length=150)
    except Exception as e:
        st.error(f"Error generating description: {str(e)}")
        return None
# Define the cached function outside the class
class OptimizedRealEstateAIApp:
    def __init__(self):
        try:
            self.tokenizer, self.model = load_or_train_models()
            self.generator = RealEstateTextGenerator(
                self.model, 
                self.tokenizer
            )
        except Exception as e:
            st.error(f"Error initializing app: {str(e)}")
            st.stop()

    def generate_property_description(self, property_type, location):
        try:
            marketing_plan = self.generator.generate_marketing_plan(
                property_type, 
                location
            )
            
            # Convert to JSON string for proper formatting
            return json.dumps(marketing_plan, ensure_ascii=False, indent=2)
            
        except Exception as e:
            st.error(f"Error generating marketing plan: {str(e)}")
            return None

    def run(self):
        try:
            # Set Arabic text direction
            st.markdown("""
                <style>
                .stApp { direction: rtl; }
                .css-18e3th9 { direction: rtl; }
                div[data-testid="stMarkdownContainer"] { direction: rtl; }
                </style>
            """, unsafe_allow_html=True)
            
            # App title
            st.title("🏢 مولد المحتوى العقاري باستخدام الذكاء الاصطناعي")
            
            # Input fields
            col1, col2 = st.columns(2)
            with col1:
                location = st.selectbox(
                    "المدينة",
                    ["الرياض", "جدة", "الدمام"],
                    index=None,
                    placeholder="اختر المدينة"
                )
            with col2:
                property_type = st.selectbox(
                    "نوع العقار",
                    ["شقة سكنية", "فيلا", "مجمع سكني"],
                    index=None,
                    placeholder="اختر نوع العقار"
                )
            
            # Generate button
            if st.button("توليد خطة تسويقية"):
                if not location or not property_type:
                    st.warning("الرجاء اختيار المدينة ونوع العقار")
                    return
                    
                with st.spinner('جاري إنشاء الخطة التسويقية...'):
                    marketing_plan = self.generate_property_description(
                        property_type, 
                        location
                    )
                    if marketing_plan:
                        st.markdown("### الخطة التسويقية")
                        st.json(marketing_plan)
                        
                        # Add download button
                        st.download_button(
                            "تحميل الخطة التسويقية",
                            marketing_plan,
                            f"marketing_plan_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                            mime="application/json"
                        )
                    
        except Exception as e:
            logger.error(f"Error in app execution: {str(e)}")
            st.error(f"حدث خطأ: {str(e)}")

if __name__ == "__main__":
    try:
        app = OptimizedRealEstateAIApp()
        app.run()
        
    except Exception as e:
        st.error(f"Application error: {str(e)}")
        logger.error(f"Application error: {str(e)}")
