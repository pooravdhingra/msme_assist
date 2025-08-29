# msme_bot.py
import logging
import time
import asyncio
import json
import hashlib
from datetime import datetime, timedelta
from dataclasses import dataclass
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA
from data_loader import load_rag_data, load_dfl_data, PineconeRecordRetriever
from scheme_lookup import (
    DocumentListRetriever,
)
from scheme_mongo_lookup import (
    initialize_mongo_scheme_retriever,
    find_scheme_guid_by_query_mongo as find_scheme_guid_by_query,
    fetch_scheme_docs_by_guid_mongo as fetch_scheme_docs_by_guid,
    search_schemes_by_query_mongo as search_schemes_by_query,
)
from utils import extract_scheme_guid
from data import DataManager
import re
import os
from functools import lru_cache
from concurrent.futures import ThreadPoolExecutor
from typing import Optional, Dict, Any, AsyncGenerator, Tuple , Set
import redis.asyncio as redis

# Set up logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    force=True)
logger = logging.getLogger(__name__)
logging.getLogger("pymongo").setLevel(logging.WARNING)

# Load environment variables
load_dotenv()

# Thread pool for CPU-intensive operations
executor = ThreadPoolExecutor(max_workers=8)

WELCOME_MESSAGES = {
    "en": {
        "full": "नमस्ते {user_name}! हकदर्शक में स्वागत है। मैं यहाँ आपको {state_name} और केंद्रीय योजनाओं के लिए सरकारी योजनाएं और दस्तावेज़ खोजने में मदद करने के लिए हूँ। यदि आपको डिजिटल कौशल, वित्तीय साक्षरता, या अपने व्यवसाय को बढ़ाने में सहायता चाहिए, तो बस पूछें। आइए मिलकर आपके व्यवसाय को सफल बनाते हैं!",
        "short": "नमस्ते {user_name}! हकदर्शक में स्वागत है। मैं आपको {state_name} और केंद्रीय योजनाओं के लिए सरकारी योजनाएं और दस्तावेज़ खोजने में मदद करूंगा।"
    },
    "hi": {
        "full": "नमस्ते {user_name}! हकदर्शक में स्वागत है। मैं यहाँ आपको {state_name} और केंद्रीय योजनाओं के लिए सरकारी योजनाएं और दस्तावेज़ खोजने में मदद करने के लिए हूँ। यदि आपको डिजिटल कौशल, वित्तीय साक्षरता, या अपने व्यवसाय को बढ़ाने में सहायता चाहिए, तो बस पूछें। आइए मिलकर आपके व्यवसाय को सफल बनाते हैं!",
        "short": "नमस्ते {user_name}! हकदर्शक में स्वागत है। मैं आपको {state_name} और केंद्रीय योजनाओं के लिए सरकारी योजनाएं और दस्तावेज़ खोजने में मदद करूंगा।"
    },
    "mr": {
        "full": "नमस्कार {user_name}! हकदर्शक मध्ये आपले स्वागत आहे. मी येथे तुम्हाला {state_name} आणि केंद्र सरकारच्या योजनांसाठी सरकारी योजना आणि कागदपत्रे शोधण्यात मदत करण्यासाठी आहे. तुम्हाला डिजिटल कौशल्य, आर्थिक साक्षरता किंवा व्यवसाय वाढवण्यासाठी मदत हवी असल्यास, फक्त विचारा. चला मिळून तुमचा व्यवसाय यशस्वी करूया!",
        "short": "नमस्कार {user_name}! हकदर्शक मध्ये आपले स्वागत आहे. मी तुम्हाला {state_name} आणि केंद्र सरकारच्या योजनांसाठी सरकारी योजना आणि कागदपत्रे शोधण्यात मदत करेन."
    },
    "ta": {
        "full": "வணக்கம் {user_name}! ஹக்‌தர்ஷக்-க்கு வரவேற்கிறோம். {state_name} மற்றும் மத்திய அரசுத் திட்டங்களுக்கான அரசு திட்டங்கள் மற்றும் ஆவணங்களை நீங்கள் கண்டறிய நான் இங்கு இருக்கிறேன். டிஜிட்டல் திறன்கள், நிதி அறிவியல் அல்லது உங்கள் வணிகத்தை வளர்ப்பதில் உதவி தேவைப்பட்டால், கேளுங்கள். உங்கள் வணிகத்தை வெற்றிகரமாக மாற்ற ஒன்றாக வேலை செய்வோம்!",
        "short": "வணக்கம் {user_name}! ஹக்‌தர்ஷக்-க்கு வரவேற்கிறோம். நான் {state_name} மற்றும் மத்திய அரசுத் திட்டங்களுக்கான அரசு திட்டங்கள் மற்றும் ஆவணங்களை கண்டறிய உதவுவேன்."
    },
    "te": {
        "full": "నమస్కారం {user_name}! హక్దర్షక్‌కి స్వాగతం. నేను మీకు {state_name} మరియు కేంద్ర ప్రభుత్వ పథకాల కోసం ప్రభుత్వ పథకాలు మరియు పత్రాలను కనుగొనడంలో సహాయం చేయడానికి ఇక్కడ ఉన్నాను. డిజిటల్ నైపుణ్యాలు, ఆర్థిక సాక్షరత లేదా మీ వ్యాపారాన్ని పెంచుకోవడానికి సహాయం అవసరమైతే, కేవలం అడగండి. మనం కలిసి మీ వ్యాపారాన్ని విజయవంతం చేద్దాం!",
        "short": "నమస్కారం {user_name}! హక్దర్షక్‌కి స్వాగతం. నేను మీకు {state_name} మరియు కేంద్ర ప్రభుత్వ పథకాల కోసం ప్రభుత్వ పథకాలు మరియు పత్రాలను కనుగొనడంలో సహాయం చేస్తాను."
    },
    "bn": {
        "full": "নমস্কার {user_name}! হকদর্শকে আপনাকে স্বাগতম। আমি এখানে {state_name} এবং কেন্দ্রীয় প্রকল্পগুলির জন্য সরকারি প্রকল্প এবং নথি খুঁজে পেতে আপনাকে সাহায্য করতে এসেছি। যদি আপনার ডিজিটাল দক্ষতা, আর্থিক সাক্ষরতা বা ব্যবসা বৃদ্ধি সম্পর্কে সহায়তা প্রয়োজন হয়, তবে জিজ্ঞাসা করুন। আসুন একসাথে আপনার ব্যবসাকে সফল করি!",
        "short": "নমস্কার {user_name}! হকদর্শকে আপনাকে স্বাগতম। আমি {state_name} এবং কেন্দ্রীয় প্রকল্পগুলির জন্য সরকারি প্রকল্প এবং নথি খুঁজে পেতে আপনাকে সাহায্য করব।"
    },
    "gu": {
        "full": "નમસ્તે {user_name}! હકદરશકમાં આપનું સ્વાગત છે. હું અહીં {state_name} અને કેન્દ્રીય યોજનાઓ માટે સરકારી યોજનાઓ અને દસ્તાવેજો શોધવામાં તમને મદદ કરવા માટે છું. જો તમને ડિજિટલ કૌશલ્ય, નાણાકીય સાક્ષરતા અથવા તમારા વ્યવસાયને વધારવામાં મદદ જોઈએ, તો ફક્ત પૂછો. ચાલો મળીને તમારો વ્યવસાય સફળ બનાવીએ!",
        "short": "નમસ્તે {user_name}! હકદરશકમાં આપનું સ્વાગત છે. હું {state_name} અને કેન્દ્રીય યોજનાઓ માટે સરકારી યોજનાઓ અને દસ્તાવેજો શોધવામાં તમારી મદદ કરીશ."
    },
    "kn": {
        "full": "ನಮಸ್ಕಾರ {user_name}! ಹಕ್‌ದರ್ಶಕ್‌ಗೆ ಸುಸ್ವಾಗತ. ನಾನು ಇಲ್ಲಿ {state_name} ಮತ್ತು ಕೇಂದ್ರ ಸರ್ಕಾರದ ಯೋಜನೆಗಳಿಗೆ ಸರ್ಕಾರಿ ಯೋಜನೆಗಳು ಮತ್ತು ದಾಖಲೆಗಳನ್ನು ಹುಡುಕಲು ನಿಮಗೆ ಸಹಾಯ ಮಾಡಲು ಬಂದಿದ್ದೇನೆ. ಡಿಜಿಟಲ್ ಕೌಶಲ್ಯಗಳು, ಹಣಕಾಸು ಪಾಠಶಾಲೆ ಅಥವಾ ನಿಮ್ಮ ವ್ಯವಹಾರವನ್ನು ವೃದ್ಧಿಸಲು ಸಹಾಯ ಬೇಕಾದರೆ, ಕೇಳಿ. ನಾವು ಒಟ್ಟಾಗಿ ನಿಮ್ಮ ವ್ಯವಹಾರವನ್ನು ಯಶಸ್ವಿಗೊಳಿಸೋಣ!",
        "short": "ನಮಸ್ಕಾರ {user_name}! ಹಕ್‌ದರ್ಶಕ್‌ಗೆ ಸುಸ್ವಾಗತ. ನಾನು {state_name} ಮತ್ತು ಕೇಂದ್ರ ಸರ್ಕಾರದ ಯೋಜನೆಗಳಿಗೆ ಸರ್ಕಾರಿ ಯೋಜನೆಗಳು ಮತ್ತು ದಾಖಲೆಗಳನ್ನು ಹುಡುಕಲು ನಿಮಗೆ ಸಹಾಯ ಮಾಡುತ್ತೇನೆ."
    },
    "or": {
        "full": "ନମସ୍କାର {user_name}! ହକଦର୍ଶକରେ ଆପଣଙ୍କୁ ସ୍ବାଗତ। ମୁଁ {state_name} ଏବଂ କେନ୍ଦ୍ରୀୟ ଯୋଜନାଗୁଡିକ ପାଇଁ ସରକାରୀ ଯୋଜନା ଏବଂ ଦଳିଲଗୁଡିକ ଖୋଜିବାରେ ଆପଣଙ୍କୁ ସହଯୋଗ କରିବି। ଯଦି ଆପଣଙ୍କୁ ଡିଜିଟାଲ୍ କୌଶଳ, ଆର୍ଥିକ ସାକ୍ଷରତା କିମ୍ବା ବ୍ୟବସାୟ ବୃଦ୍ଧିରେ ସହାୟତା ଆବଶ୍ୟକ, ଦୟାକରି ପଚାରନ୍ତୁ। ଆସନ୍ତୁ ଆମେ ମିଳିତଭାବେ ଆପଣଙ୍କ ବ୍ୟବସାୟକୁ ସଫଳ କରିବା!",
        "short": "ନମସ୍କାର {user_name}! ହକଦର୍ଶକରେ ଆପଣଙ୍କୁ ସ୍ବାଗତ। ମୁଁ {state_name} ଏବଂ କେନ୍ଦ୍ରୀୟ ଯୋଜନାଗୁଡିକ ପାଇଁ ସରକାରୀ ଯୋଜନା ଏବଂ ଦଳିଲଗୁଡିକ ଖୋଜିବାରେ ଆପଣଙ୍କୁ ସହଯୋଗ କରିବି।"
    },
    "ml": {
        "full": "നമസ്കാരം {user_name}! ഹക്ദർശക്-ലേക്ക് സ്വാഗതം. ഞാൻ ഇവിടെ {state_name} സംസ്ഥാനത്തിനും കേന്ദ്ര പദ്ധതികൾക്കുമുള്ള സർക്കാർ പദ്ധതികളും രേഖകളും കണ്ടെത്തുന്നതിൽ നിങ്ങളെ സഹായിക്കാൻ എത്തി. നിങ്ങൾക്ക് ഡിജിറ്റൽ കഴിവുകൾ, സാമ്പത്തിക വിജ്ഞാനം അല്ലെങ്കിൽ നിങ്ങളുടെ ബിസിനസ്സ് വളർത്തുന്നതിനുള്ള സഹായം ആവശ്യമുണ്ടെങ്കിൽ, ചോദിക്കുക. നമുക്ക് ഒന്നിച്ച് നിങ്ങളുടെ ബിസിനസ്സ് വിജയകരമാക്കാം!",
        "short": "നമസ്കാരം {user_name}! ഹക്ദർശക്-ലേക്ക് സ്വാഗതം. ഞാൻ {state_name} സംസ്ഥാനത്തിനും കേന്ദ്ര പദ്ധതികൾക്കുമുള്ള സർക്കാർ പദ്ധതികളും രേഖകളും കണ്ടെത്തുന്നതിൽ നിങ്ങളെ സഹായിക്കും."
    },
    "pa": {
        "full": "ਸਤ ਸ੍ਰੀ ਅਕਾਲ {user_name}! ਹਕਦਰਸ਼ਕ ਵਿੱਚ ਤੁਹਾਡਾ ਸਵਾਗਤ ਹੈ। ਮੈਂ ਤੁਹਾਨੂੰ {state_name} ਅਤੇ ਕੇਂਦਰੀ ਯੋਜਨਾਵਾਂ ਲਈ ਸਰਕਾਰੀ ਯੋਜਨਾਵਾਂ ਅਤੇ ਦਸਤਾਵੇਜ਼ ਲੱਭਣ ਵਿੱਚ ਮਦਦ ਕਰਨ ਲਈ ਇੱਥੇ ਹਾਂ। ਜੇ ਤੁਹਾਨੂੰ ਡਿਜ਼ੀਟਲ ਹੁਨਰਾਂ, ਵਿੱਤੀ ਸਾਖਰਤਾ ਜਾਂ ਆਪਣੇ ਕਾਰੋਬਾਰ ਨੂੰ ਵਧਾਉਣ ਵਿੱਚ ਸਹਾਇਤਾ ਦੀ ਲੋੜ ਹੈ, ਤਾਂ ਬੇਝਿਝਕ ਪੁੱਛੋ। ਆਓ ਮਿਲ ਕੇ ਤੁਹਾਡੇ ਕਾਰੋਬਾਰ ਨੂੰ ਸਫਲ ਬਣਾਈਏ!",
        "short": "ਸਤ ਸ੍ਰੀ ਅਕਾਲ {user_name}! ਹਕਦਰਸ਼ਕ ਵਿੱਚ ਤੁਹਾਡਾ ਸਵਾਗਤ ਹੈ। ਮੈਂ ਤੁਹਾਨੂੰ {state_name} ਅਤੇ ਕੇਂਦਰੀ ਯੋਜਨਾਵਾਂ ਲਈ ਸਰਕਾਰੀ ਯੋਜਨਾਵਾਂ ਅਤੇ ਦਸਤਾਵੇਜ਼ ਲੱਭਣ ਵਿੱਚ ਮਦਦ ਕਰਾਂਗਾ।"
    },
    "as": {
        "full": "নমস্কাৰ {user_name}! হকদৰ্ছকত আপোনাক স্বাগতম। মই ইয়াত {state_name} আৰু কেন্দ্ৰীয় আঁচনিৰ বাবে চৰকাৰী আঁচনি আৰু নথিপত্ৰ বিচাৰি পাবলৈ আপোনাক সহায় কৰিবলৈ আহিছো। যদি আপোনাৰ ডি়জিটেল দক্ষতা, আৰ্থিক সাক্ষৰতা বা ব্যৱসায় বৃদ্ধিৰ প্ৰয়োজন হয়, তেন্তে সোধক। আহক, আমি একেলগে আপোনাৰ ব্যৱসায় সফল কৰোঁ!",
        "short": "নমস্কাৰ {user_name}! হকদৰ্ছকত আপোনাক স্বাগতম। মই {state_name} আৰু কেন্দ্ৰীয় আঁচনিৰ বাবে চৰকাৰী আঁচনি আৰু নথিপত্ৰ বিচাৰি পাবলৈ আপোনাক সহায় কৰিম।"
    },
    "ur": {
        "full": "السلام علیکم {user_name}! حق درشک میں خوش آمدید۔ میں یہاں {state_name} اور مرکزی اسکیموں کے لئے حکومتی اسکیمیں اور دستاویزات تلاش کرنے میں آپ کی مدد کے لئے موجود ہوں۔ اگر آپ کو ڈیجیٹل مہارت، مالی خواندگی، یا اپنے کاروبار کو بڑھانے میں مدد چاہیے تو بس پوچھیے۔ آئیے مل کر آپ کے کاروبار کو کامیاب بناتے ہیں!",
        "short": "السلام علیکم {user_name}! حق درشک میں خوش آمدید۔ میں {state_name} اور مرکزی اسکیموں کے لئے حکومتی اسکیمیں اور دستاویزات تلاش کرنے میں آپ کی مدد کروں گا۔"
    }
}

LANGUAGE_LABELS = {
    "en": "English",
    "mr": "Marathi", 
    "hi": "Hindi",
    "as": "Assamese",
    "te": "Telugu",
    "bn": "Bengali", 
    "ta": "Tamil",
    "gu": "Gujarati",
    "kn": "Kannada",
    "or": "Odia",
    "ml": "Malayalam",
    "pa": "Punjabi",
    "ur": "Urdu"
}

# Language-specific messages and greetings
LANGUAGE_CONFIG = {
    "English":{
        "greeting": "नमस्ते",
        "out_of_scope": "क्षमा करें, मैं केवल सरकारी योजनाओं, डिजिटल या वित्तीय साक्षरता और व्यावसायिक वृद्धि पर मदद कर सकता हूँ।",
        "gratitude_fallback": "धन्यवाद! क्या मैं और मदद कर सकता हूँ?",
        "no_info": "मुझे इसके बारे में अभी जानकारी नहीं है।",
        "rag_search": "मुझे [scheme name] के बारे में और जानकारी लेनी होगी। क्या आप इसी योजना की बात कर रहे हैं?",
        "eligibility_question": "पात्रता या आवेदन करने के बारे में जानना चाहते हैं?",
        "scheme_question": "किसी योजना के बारे में और जानना चाहते हैं?",
        "which_scheme": "कौन सी योजना के बारे में?",
        "error_message": "क्षमा करें, मैं आपका प्रश्न संसाधित नहीं कर सका।",
        "haqdarshak_message": "हकदर्शक आपको यह दस्तावेज़ दिलाने में मदद कर सकता है। कृपया ऐप में बुक करें।"
    },
    "Hindi": {
        "greeting": "नमस्ते",
        "out_of_scope": "क्षमा करें, मैं केवल सरकारी योजनाओं, डिजिटल या वित्तीय साक्षरता और व्यावसायिक वृद्धि पर मदद कर सकता हूँ।",
        "gratitude_fallback": "धन्यवाद! क्या मैं और मदद कर सकता हूँ?",
        "no_info": "मुझे इसके बारे में अभी जानकारी नहीं है।",
        "rag_search": "मुझे [scheme name] के बारे में और जानकारी लेनी होगी। क्या आप इसी योजना की बात कर रहे हैं?",
        "eligibility_question": "पात्रता या आवेदन करने के बारे में जानना चाहते हैं?",
        "scheme_question": "किसी योजना के बारे में और जानना चाहते हैं?",
        "which_scheme": "कौन सी योजना के बारे में?",
        "error_message": "क्षमा करें, मैं आपका प्रश्न संसाधित नहीं कर सका।",
        "haqdarshak_message": "हकदर्शक आपको यह दस्तावेज़ दिलाने में मदद कर सकता है। कृपया ऐप में बुक करें।"
    },
    "Tamil": {
        "greeting": "வணக்கம்",
        "out_of_scope": "மன்னிக்கவும், நான் அரசு திட்டங்கள், டிஜிட்டல் அல்லது நிதி கல்வியறிவு மற்றும் வணிக வளர்ச்சியில் மட்டுமே உதவ முடியும்.",
        "gratitude_fallback": "நன்றி! வேறு ஏதாவது உதவி தேவையா?",
        "no_info": "இதைப் பற்றி எனக்கு இப்போது தகவல் இல்லை.",
        "rag_search": "[scheme name] பற்றி மேலும் தகவல் பெற வேண்டும். நீங்கள் இந்த திட்டத்தைப் பற்றி பேசுகிறீர்களா?",
        "eligibility_question": "தகுதி அல்லது விண்ணப்பிப்பது எப்படி என்று தெரிந்து கொள்ள விரும்புகிறீர்களா?",
        "scheme_question": "ஏதேனும் திட்டத்தைப் பற்றி மேலும் அறிய விரும்புகிறீர்களா?",
        "which_scheme": "எந்த திட்டத்தைப் பற்றி கேட்கிறீர்கள்?",
        "error_message": "மன்னிக்கவும், உங்கள் கேள்வியை செயலாக்க முடியவில்லை.",
        "haqdarshak_message": "ஹக்தர்ஷக் இந்த ஆவணத்தை பெற உங்களுக்கு உதவ முடியும். தயவுசெய்து ஆப்பில் புக் செய்யுங்கள்."
    },
    "Telugu": {
        "greeting": "నమస్కారం",
        "out_of_scope": "క్షమించండి, నేను ప్రభుత్వ పథకాలు, డిజిటల్ లేదా ఆర్థిక అక్షరాస్యత మరియు వ్యాపార వృద్ధిలో మాత్రమే సహాయం చేయగలను.",
        "gratitude_fallback": "ధన్యవాదాలు! నేను మరేమైనా సహాయం చేయగలనా?",
        "no_info": "దీని గురించి నాకు ఇప్పుడు సమాచారం లేదు.",
        "rag_search": "[scheme name] గురించి మరింత సమాచారం అవసరం. మీరు ఈ పథకం గురించి మాట్లాడుతున్నారా?",
        "eligibility_question": "అర్హత లేదా దరఖాస్తు ఎలా చేయాలో తెలుసుకోవాలనుకుంటున్నారా?",
        "scheme_question": "ఏదైనా పథకం గురించి మరింత తెలుసుకోవాలనుకుంటున్నారా?",
        "which_scheme": "ఏ పథకం గురించి అడుగుతున్నారు?",
        "error_message": "క్షమించండి, మీ ప్రశ్నను ప్రాసెస్ చేయలేకపోయాను.",
        "haqdarshak_message": "ఈ డాక్యుమెంట్‌ను పొందడంలో హక్‌దర్శక్ మీకు సహాయం చేయగలదు. దయచేసి యాప్‌లో బుక్ చేయండి."
    },
    "Marathi": {
        "greeting": "नमस्कार",
        "out_of_scope": "क्षमस्व, मी फक्त सरकारी योजना, डिजिटल किंवा आर्थिक साक्षरता आणि व्यावसायिक वाढीत मदत करू शकतो.",
        "gratitude_fallback": "धन्यवाद! मी आणखी काही मदत करू का?",
        "no_info": "याबद्दल मला आता माहिती नाही.",
        "rag_search": "[scheme name] बद्दल अधिक माहिती घ्यावी लागेल. तुम्ही या योजनेबद्दल बोलत आहात का?",
        "eligibility_question": "पात्रता किंवा अर्ज कसा करावा हे जाणून घ्यायचे आहे का?",
        "scheme_question": "कोणत्याही योजनेबद्दल अधिक जाणून घ्यायचे आहे का?",
        "which_scheme": "कोणत्या योजनेबद्दल विचारत आहात?",
        "error_message": "क्षमस्व, मी तुमचा प्रश्न प्रक्रिया करू शकलो नाही.",
        "haqdarshak_message": "हक्दर्शक तुम्हाला हा कागदपत्र मिळवून देण्यात मदत करू शकतो. कृपया अॅपमध्ये बुक करा."
    },
    "Bengali": {
        "greeting": "নমস্কার",
        "out_of_scope": "দুঃখিত, আমি শুধুমাত্র সরকারি প্রকল্প, ডিজিটাল বা আর্থিক সাক্ষরতা এবং ব্যবসায়িক বৃদ্ধিতে সাহায্য করতে পারি।",
        "gratitude_fallback": "ধন্যবাদ! আমি আর কিছুতে সাহায্য করতে পারি?",
        "no_info": "এই বিষয়ে আমার এখন তথ্য নেই।",
        "rag_search": "[scheme name] সম্পর্কে আরও তথ্য নিতে হবে। আপনি কি এই প্রকল্পের কথা বলছেন?",
        "eligibility_question": "যোগ্যতা বা আবেদন করার বিষয়ে জানতে চান?",
        "scheme_question": "কোনো প্রকল্প সম্পর্কে আরও জানতে চান?",
        "which_scheme": "কোন প্রকল্প সম্পর্কে জিজ্ঞাসা করছেন?",
        "error_message": "দুঃখিত, আমি আপনার প্রশ্ন প্রক্রিয়া করতে পারিনি।",
        "haqdarshak_message": "হকদর্শক আপনাকে এই নথি পেতে সাহায্য করতে পারে। অ্যাপে বুক করুন।"
    },
    "Assamese": {
        "greeting": "নমস্কাৰ",
        "out_of_scope": "দুঃখিত, মই কেৱল চৰকাৰী আঁচনি, ডিজিটেল বা আৰ্থিক সাক্ষৰতা আৰু ব্যৱসায়িক বৃদ্ধিত সহায় কৰিব পাৰোঁ।",
        "gratitude_fallback": "ধন্যবাদ! মই আৰু কিবা সহায় কৰিব পাৰোঁ নেকি?",
        "no_info": "এই বিষয়ে মোৰ এতিয়া তথ্য নাই।",
        "rag_search": "[scheme name] সম্পৰ্কে অধিক তথ্য লাগিব। আপুনি এই আঁচনিৰ কথা কৈছে নেকি?",
        "eligibility_question": "যোগ্যতা বা আবেদন কৰাৰ বিষয়ে জানিব খোজে নেকি?",
        "scheme_question": "কোনো আঁচনিৰ বিষয়ে অধিক জানিব খোজে নেকি?",
        "which_scheme": "কোন আঁচনিৰ বিষয়ে সুধিছে?",
        "error_message": "দুঃখিত, মই আপোনাৰ প্ৰশ্ন প্ৰক্ৰিয়া কৰিব নোৱাৰিলোঁ।",
        "haqdarshak_message": "হকদৰ্শকে আপোনাক এই দস্তাবেজ পাবলৈ সহায় কৰিব পাৰে। এপত বুক কৰক।"
    },
    "Gujarati": {
        "greeting": "નમસ્તે",
        "out_of_scope": "માફ કરશો, હું ફક્ત સરકારી યોજનાઓ, ડિજિટલ અથવા નાણાકીય સાક્ષરતા અને વ્યાવસાયિક વૃદ્ધિમાં મદદ કરી શકું છું.",
        "gratitude_fallback": "આભાર! હું બીજું કંઈ મદદ કરી શકું?",
        "no_info": "આ વિશે મારી પાસે હમણાં માહિતી નથી.",
        "rag_search": "[scheme name] વિશે વધુ માહિતી લેવી પડશે. તમે આ યોજના વિશે વાત કરો છો?",
        "eligibility_question": "પાત્રતા અથવા અરજી કરવા વિશે જાણવા માંગો છો?",
        "scheme_question": "કોઈ યોજના વિશે વધુ જાણવા માંગો છો?",
        "which_scheme": "કઈ યોજના વિશે પૂછો છો?",
        "error_message": "માફ કરશો, હું તમારા પ્રશ્નનો પ્રોસેસ કરી શક્યો નહીં.",
        "haqdarshak_message": "હકદર્શક તમને આ દસ્તાવેજ મેળવવામાં મદદ કરી શકે છે. કૃપા કરીને એપમાં બુક કરો."
    },
    "Kannada": {
        "greeting": "ನಮಸ್ಕಾರ",
        "out_of_scope": "ಕ್ಷಮಿಸಿ, ನಾನು ಕೇವಲ ಸರ್ಕಾರಿ ಯೋಜನೆಗಳು, ಡಿಜಿಟಲ್ ಅಥವಾ ಹಣಕಾಸಿನ ಸಾಕ್ಷರತೆ ಮತ್ತು ವ್ಯಾಪಾರ ಬೆಳವಣಿಗೆಯಲ್ಲಿ ಸಹಾಯ ಮಾಡಬಹುದು.",
        "gratitude_fallback": "ಧನ್ಯವಾದಗಳು! ನಾನು ಬೇರೆ ಏನಾದರೂ ಸಹಾಯ ಮಾಡಬಹುದೇ?",
        "no_info": "ಇದರ ಬಗ್ಗೆ ನನಗೆ ಈಗ ಮಾಹಿತಿ ಇಲ್ಲ.",
        "rag_search": "[scheme name] ಬಗ್ಗೆ ಹೆಚ್ಚಿನ ಮಾಹಿತಿ ಬೇಕಾಗಿದೆ. ನೀವು ಈ ಯೋಜನೆಯ ಬಗ್ಗೆ ಮಾತನಾಡುತ್ತಿದ್ದೀರಾ?",
        "eligibility_question": "ಅರ್ಹತೆ ಅಥವಾ ಅರ್ಜಿ ಸಲ್ಲಿಸುವ ಬಗ್ಗೆ ತಿಳಿಯಲು ಬಯಸುವಿರಾ?",
        "scheme_question": "ಯಾವುದೇ ಯೋಜನೆಯ ಬಗ್ಗೆ ಹೆಚ್ಚು ತಿಳಿಯಲು ಬಯಸುವಿರಾ?",
        "which_scheme": "ಯಾವ ಯೋಜನೆಯ ಬಗ್ಗೆ ಕೇಳುತ್ತಿದ್ದೀರಿ?",
        "error_message": "ಕ್ಷಮಿಸಿ, ನಾನು ನಿಮ್ಮ ಪ್ರಶ್ನೆಯನ್ನು ಪ್ರಕ್ರಿಯೆಗೊಳಿಸಲು ಸಾಧ್ಯವಾಗಲಿಲ್ಲ.",
        "haqdarshak_message": "ಈ ದಾಖಲೆಯನ್ನು ಪಡೆಯಲು ಹಕ್‌ದರ್ಶಕ್ ನಿಮಗೆ ಸಹಾಯ ಮಾಡಬಹುದು. ದಯವಿಟ್ಟು ಆ್ಯಪ್‌ನಲ್ಲಿ ಬುಕ್ ಮಾಡಿ."
    },
    "Odia": {
        "greeting": "ନମସ୍କାର",
        "out_of_scope": "ଦୁଃଖିତ, ମୁଁ କେବଳ ସରକାରୀ ଯୋଜନା, ଡିଜିଟାଲ କିମ୍ବା ଆର୍ଥିକ ସାକ୍ଷରତା ଏବଂ ବ୍ୟବସାୟିକ ବୃଦ୍ଧିରେ ସାହାଯ୍ୟ କରିପାରିବି।",
        "gratitude_fallback": "ଧନ୍ୟବାଦ! ମୁଁ ଆଉ କିଛି ସାହାଯ୍ୟ କରିପାରିବି କି?",
        "no_info": "ଏହା ବିଷୟରେ ମୋର ବର୍ତ୍ତମାନ ତଥ୍ୟ ନାହିଁ।",
        "rag_search": "[scheme name] ବିଷୟରେ ଅଧିକ ତଥ୍ୟ ଦରକାର। ଆପଣ ଏହି ଯୋଜନା ବିଷୟରେ କହୁଛନ୍ତି କି?",
        "eligibility_question": "ଯୋଗ୍ୟତା କିମ୍ବା ଆବେଦନ କରିବା ବିଷୟରେ ଜାଣିବାକୁ ଚାହାଁନ୍ତି କି?",
        "scheme_question": "କୌଣସି ଯୋଜନା ବିଷୟରେ ଅଧିକ ଜାଣିବାକୁ ଚାହାଁନ୍ତି କି?",
        "which_scheme": "କେଉଁ ଯୋଜନା ବିଷୟରେ ପଚାରୁଛନ୍ତି?",
        "error_message": "ଦୁଃଖିତ, ମୁଁ ଆପଣଙ୍କ ପ୍ରଶ୍ନ ପ୍ରକ୍ରିୟାକରଣ କରିପାରିଲି ନାହିଁ।",
        "haqdarshak_message": "ଏହି ଡକୁମେଣ୍ଟ ପାଇବାରେ ହକଦର୍ଶକ ଆପଣଙ୍କୁ ସାହାଯ୍ୟ କରିପାରିବ। ଦୟାକରି ଆପରେ ବୁକ କରନ୍ତୁ।"
    },
    "Malayalam": {
        "greeting": "നമസ്കാരം",
        "out_of_scope": "ക്ഷമിക്കണം, എനിക്ക് സർക്കാർ പദ്ധതികൾ, ഡിജിറ്റൽ അല്ലെങ്കിൽ സാമ്പത്തിക സാക്ഷരता, ബിസിനസ് വളർച്ച എന്നിവയിൽ മാത്രമേ സഹായിക്കാൻ കഴിയൂ.",
        "gratitude_fallback": "നന്ദി! എനിക്ക് മറ്റെന്തെങ്കിലും സഹായിക്കാൻ കഴിയുമോ?",
        "no_info": "ഇതിനെക്കുറിച്ച് എനിക്ക് ഇപ്പോൾ വിവരങ്ങളില്ല.",
        "rag_search": "[scheme name] നെക്കുറിച്ച് കൂടുതൽ വിവരങ്ങൾ ആവശ്യമാണ്. നിങ്ങൾ ഈ പദ്ധതിയെക്കുറിച്ചാണോ പറയുന്നത്?",
        "eligibility_question": "യോഗ്യത അല്ലെങ്കിൽ അപേക്ഷിക്കുന്നതിനെക്കുറിച്ച് അറിയാൻ ആഗ്രഹിക്കുന്നുണ്ടോ?",
        "scheme_question": "ഏതെങ്കിലും പദ്ധതിയെക്കുറിച്ച് കൂടുതൽ അറിയാൻ ആഗ്രഹിക്കുന്നുണ്ടോ?",
        "which_scheme": "ഏത് പദ്ധതിയെക്കുറിച്ചാണ് ചോദിക്കുന്നത്?",
        "error_message": "ക്ഷമിക്കണം, എനിക്ക് നിങ്ങളുടെ ചോദ്യം പ്രോസസ്സ് ചെയ്യാൻ കഴിഞ്ഞില്ല.",
        "haqdarshak_message": "ഈ ഡോക്യുമെന്റ് ലഭിക്കുന്നതിന് ഹക്ദർശക് നിങ്ങളെ സഹായിക്കും. ആപ്പിൽ ബുക്ക് ചെയ്യുക."
    },
    "Punjabi": {
        "greeting": "ਸਤ ਸ੍ਰੀ ਅਕਾਲ",
        "out_of_scope": "ਮਾਫ਼ ਕਰਨਾ, ਮੈਂ ਕੇਵਲ ਸਰਕਾਰੀ ਯੋਜਨਾਵਾਂ, ਡਿਜੀਟਲ ਜਾਂ ਵਿੱਤੀ ਸਾਖਰਤਾ ਅਤੇ ਵਪਾਰਕ ਵਿਕਾਸ ਵਿੱਚ ਮਦਦ ਕਰ ਸਕਦਾ ਹਾਂ।",
        "gratitude_fallback": "ਧੰਨਵਾਦ! ਮੈਂ ਹੋਰ ਕੁਝ ਮਦਦ ਕਰ ਸਕਦਾ ਹਾਂ?",
        "no_info": "ਇਸ ਬਾਰੇ ਮੇਰੇ ਕੋਲ ਹੁਣ ਜਾਣਕਾਰੀ ਨਹੀਂ ਹੈ।",
        "rag_search": "[scheme name] ਬਾਰੇ ਹੋਰ ਜਾਣਕਾਰੀ ਚਾਹੀਦੀ ਹੈ। ਕੀ ਤੁਸੀਂ ਇਸ ਯੋਜਨਾ ਬਾਰੇ ਗੱਲ ਕਰ ਰਹੇ ਹੋ?",
        "eligibility_question": "ਯੋਗਤਾ ਜਾਂ ਅਰਜ਼ੀ ਦੇਣ ਬਾਰੇ ਜਾਣਨਾ ਚਾਹੁੰਦੇ ਹੋ?",
        "scheme_question": "ਕਿਸੇ ਯੋਜਨਾ ਬਾਰੇ ਹੋਰ ਜਾਣਨਾ ਚਾਹੁੰਦੇ ਹੋ?",
        "which_scheme": "ਕਿਸ ਯੋਜਨਾ ਬਾਰੇ ਪੁੱਛ ਰਹੇ ਹੋ?",
        "error_message": "ਮਾਫ਼ ਕਰਨਾ, ਮੈਂ ਤੁਹਾਡੇ ਸਵਾਲ ਦਾ ਪ੍ਰੋਸੈਸ ਨਹੀਂ ਕਰ ਸਕਿਆ।",
        "haqdarshak_message": "ਇਹ ਦਸਤਾਵੇਜ਼ ਪ੍ਰਾਪਤ ਕਰਨ ਲਈ ਹਕਦਰਸ਼ਕ ਤੁਹਾਡੀ ਮਦਦ ਕਰ ਸਕਦਾ ਹੈ। ਐਪ ਵਿੱਚ ਬੁੱਕ ਕਰੋ।"
    },
    "Urdu": {
        "greeting": "آداب",
        "out_of_scope": "معذرت، میں صرف حکومتی اسکیموں، ڈیجیٹل یا مالی خواندگی اور کاروباری ترقی میں مدد کر سکتا ہوں۔",
        "gratitude_fallback": "شکریہ! کیا میں اور کوئی مدد کر سکتا ہوں؟",
        "no_info": "اس کے بارے میں میرے پاس ابھی معلومات نہیں ہیں۔",
        "rag_search": "[scheme name] کے بارے میں مزید معلومات درکار ہیں۔ کیا آپ اسی اسکیم کی بات کر رہے ہیں؟",
        "eligibility_question": "اہلیت یا درخواست دینے کے بارے میں جاننا چاہتے ہیں؟",
        "scheme_question": "کسی اسکیم کے بارے میں مزید جاننا چاہتے ہیں؟",
        "which_scheme": "کس اسکیم کے بارے میں پوچھ رہے ہیں؟",
        "error_message": "معذرت، میں آپ کا سوال پروسیس نہیں کر سکا۔",
        "haqdarshak_message": "حق درشک آپ کو یہ دستاویز حاصل کرنے میں مدد کر سکتا ہے۔ براہ کرم ایپ میں بک کریں۔"
    },
    "Hinglish": {
    "greeting": "Namaste",
    "out_of_scope": "Maaf kijiye, main sirf sarkari yojanaon, digital ya financial literacy aur business growth mein madad kar sakta hoon.",
    "gratitude_fallback": "Dhanyawad! Kya main aur madad kar sakta hoon?",
    "no_info": "Mujhe iske baare mein abhi jaankari nahi hai.",
    "rag_search": "Mujhe [scheme name] ke baare mein aur jaankari leni hogi. Kya aap isi scheme ki baat kar rahe hain?",
    "eligibility_question": "Eligibility ya apply karne ke baare mein jaanna chahte hain?",
    "scheme_question": "Kisi yojana ke baare mein aur jaanna chahte hain?",
    "which_scheme": "Kaunsi scheme ke baare mein?",
    "error_message": "Sorry, main aapka query process nahi kar saka.",
    "haqdarshak_message": "Haqdarshak aapko yeh document dilaane mein madad kar sakta hai. Kripya app mein book karein."
}
}

# Enhanced Cache Manager
class CacheManager:
    def __init__(self):
        try:
            self.redis_client = redis.Redis.from_url(
                os.getenv("REDIS_URL", "redis://localhost:6379"),
                decode_responses=True
            )
        except:
            self.redis_client = None
            logger.warning("Redis not available, using memory cache only")
        
        self.memory_cache = {}
        self.max_memory_cache_size = 1000
    
    def _cleanup_memory_cache(self):
        """Keep memory cache size under control"""
        if len(self.memory_cache) > self.max_memory_cache_size:
            # Remove oldest half of entries
            items = list(self.memory_cache.items())
            items_to_keep = items[len(items)//2:]
            self.memory_cache = dict(items_to_keep)
    
    @lru_cache(maxsize=500)
    def _generate_cache_key(self, query: str, context: str = "") -> str:
        """Generate consistent cache keys"""
        key_data = f"{query}:{context}"
        return hashlib.md5(key_data.encode()).hexdigest()
    
    async def get_intent_cache(self, query: str, conversation_history: str = "") -> Optional[str]:
        """Get cached intent with memory fallback"""
        cache_key = self._generate_cache_key(query, conversation_history)
        
        # Try Redis first
        if self.redis_client:
            try:
                cached = await self.redis_client.get(f"intent:{cache_key}")
                if cached:
                    return cached
            except:
                pass
        
        # Fallback to memory cache
        return self.memory_cache.get(f"intent:{cache_key}")
    
    async def set_intent_cache(self, query: str, intent: str, conversation_history: str = "", ttl: int = 3600):
        """Set intent cache with Redis and memory"""
        cache_key = self._generate_cache_key(query, conversation_history)
        
        # Set in Redis
        if self.redis_client:
            try:
                await self.redis_client.setex(f"intent:{cache_key}", ttl, intent)
            except:
                pass
        
        # Set in memory cache
        self.memory_cache[f"intent:{cache_key}"] = intent
        self._cleanup_memory_cache()
    
    async def get_rag_cache(self, query: str, user_context: dict = None) -> Optional[dict]:
        """Get cached RAG response"""
        context_str = json.dumps(user_context or {}, sort_keys=True)
        cache_key = self._generate_cache_key(query, context_str)
        
        if self.redis_client:
            try:
                cached = await self.redis_client.get(f"rag:{cache_key}")
                return json.loads(cached) if cached else None
            except:
                pass
        
        return self.memory_cache.get(f"rag:{cache_key}")
    
    async def set_rag_cache(self, query: str, data: dict, user_context: dict = None, ttl: int = 1800):
        """Set RAG cache"""
        context_str = json.dumps(user_context or {}, sort_keys=True)
        cache_key = self._generate_cache_key(query, context_str)
        
        if self.redis_client:
            try:
                await self.redis_client.setex(f"rag:{cache_key}", ttl, json.dumps(data))
            except:
                pass
        
        # Don't store large RAG responses in memory cache
        if len(str(data)) < 10000:  # Only cache smaller responses in memory
            self.memory_cache[f"rag:{cache_key}"] = data
            self._cleanup_memory_cache()

# Initialize cache manager
cache_manager = CacheManager()


def init_mongo_scheme_manager():
    """Initialize MongoDB scheme manager"""
    try:
        success = initialize_mongo_scheme_retriever()
        if success:
            logger.info("MongoDB scheme manager initialized successfully")
            return True
        else:
            logger.error("Failed to initialize MongoDB scheme manager")
            return False
    except Exception as e:
        logger.error(f"Failed to initialize MongoDB scheme manager: {e}")
        return False

MONGO_SCHEME_AVAILABLE = init_mongo_scheme_manager()


# Initialize DataManager - make it async capable
class AsyncDataManager(DataManager):
    async def get_conversations_async(self, mobile_number: str):
        """Async version of get_conversations"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(executor, self.get_conversations, mobile_number)
    
    async def save_conversation_async(self, session_id: str, mobile_number: str, messages: list):
        """Async version of save_conversation"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(executor, self.save_conversation, session_id, mobile_number, messages)

# Initialize async data manager
data_manager = AsyncDataManager()

# Initialize cached resources with async support
@lru_cache(maxsize=1)
def init_llm():
    """Initialize the default LLM client for all tasks except intent classification."""
    logger.info("Initializing LLM client")
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY environment variable not set")
    llm = ChatOpenAI(
        model="gpt-4.1-mini-2025-04-14",
        api_key=api_key,
        base_url="https://api.openai.com/v1",
        temperature=0
    )
    logger.info("LLM initialized")
    return llm

@lru_cache(maxsize=1)
def init_intent_llm():
    """Initialize a dedicated LLM client for intent classification."""
    logger.info("Initializing Intent LLM client")
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY environment variable not set")
    intent_llm = ChatOpenAI(
        model="gpt-4.1",
        api_key=api_key,
        base_url="https://api.openai.com/v1",
        temperature=0
    )
    logger.info("Intent LLM initialized")
    return intent_llm

@lru_cache(maxsize=1)
def init_vector_store():
    logger.info("Loading vector store")
    index_host = os.getenv("PINECONE_SCHEME_HOST")
    if not index_host:
        raise ValueError("PINECONE_SCHEME_HOST environment variable not set")
    index_name = os.getenv("PINECONE_INDEX_NAME")
    try:
        vector_store = load_rag_data(host=index_host, index_name=index_name, version_file="faiss_version.txt")
    except Exception as e:
        logger.error(f"Failed to load scheme index: {e}")
        raise
    logger.info("Vector store loaded")
    return vector_store

@lru_cache(maxsize=1)
def init_dfl_vector_store():
    logger.info("Loading DFL vector store")
    google_drive_file_id = os.getenv("DFL_GOOGLE_DOC_ID")
    if not google_drive_file_id:
        raise ValueError("DFL_GOOGLE_DOC_ID environment variable not set")
    index_host = os.getenv("PINECONE_DFL_HOST")
    if not index_host:
        raise ValueError("PINECONE_DFL_HOST environment variable not set")
    index_name = os.getenv("PINECONE_DFL_INDEX_NAME")
    try:
        vector_store = load_dfl_data(google_drive_file_id, host=index_host, index_name=index_name)
    except Exception as e:
        logger.error(f"Failed to load DFL index: {e}")
        raise
    logger.info("DFL vector store loaded")
    return vector_store

llm = init_llm()
intent_llm = init_intent_llm()
scheme_vector_store = init_vector_store()
dfl_vector_store = init_dfl_vector_store()

# Keep your existing dataclasses and helper functions unchanged
@dataclass
class UserContext:
    name: str
    state_id: str
    state_name: str
    business_name: str
    business_category: str
    gender: str

class SessionData:
    """Simple container for per-session information."""
    def __init__(self, user=None):
        self.user = user
        self.messages = []
        self.rag_cache = {}
        self.dfl_rag_cache = {}

def get_user_context(session_state):
    try:
        user = session_state.user
        return UserContext(
            name=user["fname"],
            state_id=user.get("state_id", "Unknown"),
            state_name=user.get("state_name", "Unknown"),
            business_name=user.get("business_name", "Unknown"),
            business_category=user.get("business_category", "Unknown"),
            gender=user.get("gender", "Unknown"),
        )
    except AttributeError:
        logger.error("User data not found in session state")
        return None

def detect_language(query):
    """Keep your existing language detection logic"""
    devanagari_pattern = re.compile(r'[\u0900-\u097F]')
    if devanagari_pattern.search(query):
        return "Hindi"
    
    hindi_words = [
        "kya", "kaise", "ke", "mein", "hai", "kaun", "kahan", "kab",
        "batao", "sarkari", "yojana", "paise", "karobar", "dukaan", "nayi", "naye", "chahiye", "madad", "karo",
        "dikhao", "samjhao", "tarika", "aur", "arey", "bhi", "kya", "hai", "hoga", "hogi", "ho", "hoon", "magar", "lekin", "par", 
        "toh", "ab", "phir", "kuch", "thoda", "zyada", "sab", "koi", "kuchh", "aap", "tum", "main",
        "hum", "unhe", "unko", "unse", "yeh", "woh", "aisa", "aisi", "aise", "bataiye", "achha", "acha", "accha", "theek", "theekh", 
        "thik", "thikk", "idhar", "udhar", "yahan", "wahan", "waha", "bhai", "bhaiya", "bhaiyya", 
        "bhiya", "bahut", "bahot", "bohot", "bahuut", "zara", "jara", "mat", "maat", "matlab", "matlb", "fir", "phirr", "phhir", "phir", 
        "main", "aap", "aapke", "yojanaen", "liye", "kar", "sakte", "hain", "tak"
    ]
    query_lower = query.lower()
    hindi_word_count = sum(1 for word in hindi_words if word in query_lower)
    total_words = len(query_lower.split())
    
    if total_words > 0 and hindi_word_count / total_words > 0.15:
        return "Hinglish"
    
    return "English"

def get_system_prompt(language, user_name="User", word_limit=200):

    """Return tone and style instructions."""

    system_rules = f"""1. **Language Handling**:
       - The query language is provided as {language} (English, Hindi, or Hinglish).
       - For Hindi queries, respond in Devanagari script using simple, clear words suitable for micro business owners with low Hindi proficiency.
       - For English queries, respond in Devanagari script using simple, clear Hindi words suitable for micro business owners with low Hindi proficiency.
       - For Hinglish queries, use a natural mix of simple English and Hindi words in Roman script, prioritizing hindi words in the mix.
       
       2. **Response Guidelines**:
       - Scope: Only respond to queries about government schemes, digital/financial literacy, or business growth.
       - Tone and Style: Use simple, clear words, short sentences, friendly tone, relatable examples.
       - Give structured responses with formatting like bullets or headings/subheadings. Do not give long paragraphs of text.
       - Response must be <={word_limit} words.

       - Never mention agent fees unless specified in RAG Response for scheme queries.
       - Never repeat user query or bring up ambiguity in the response, proceed directly to answering.
       - Never mention technical terms like RAG, LLM, Database etc. to the user.
       - Use scheme names exactly as provided in the RAG Response without paraphrasing (underscores may be replaced with spaces).
       - Start the response with 'नमस्ते {user_name}!' for Hindi and English queries, 'Namaste {user_name}!' for Hinglish queries unless Out_of_Scope."""

    system_prompt = system_rules.format(language=language, user_name=user_name)
    return system_prompt


# Build conversation history from stored messages for intent classification
def build_conversation_history(messages):
    conversation_history = ""
    session_messages = []
    for msg in messages[-10:]:
        if msg["role"] == "assistant" and "Welcome" in msg["content"]:
            continue
        session_messages.append((msg["role"], msg["content"], msg["timestamp"]))
    session_messages = sorted(session_messages, key=lambda x: x[2], reverse=True)[:5]
    for role, content, _ in session_messages:
        conversation_history += f"{role.capitalize()}: {content}\n"
    return conversation_history

def get_welcome_message(state_name,user_name, query_language, user_type):
    # Get messages for language or fallback to English
    messages = WELCOME_MESSAGES.get(query_language, WELCOME_MESSAGES["en"])
    template = messages["full"] if user_type == 1 else messages["short"]
    return template.format(user_name=user_name, state_name=state_name)

def welcome_user(state_name, user_name, query_language, user_type):
    """Generate a welcome message in the user's chosen language."""
    
    if query_language  in ["hi", "en", "Hinglish"]:
        query_language = "Hindi"

    if user_type == 1:
        if query_language == "Hindi":
            return f"नमस्ते {user_name}! हकदर्शक में स्वागत है। मैं यहाँ आपको {state_name} और केंद्रीय योजनाओं के लिए सरकारी योजनाएं और दस्तावेज़ खोजने में मदद करने के लिए हूँ। यदि आपको डिजिटल कौशल, वित्तीय साक्षरता, या अपने व्यवसाय को बढ़ाने में सहायता चाहिए, तो बस पूछें। आइए मिलकर आपके व्यवसाय को सफल बनाते हैं!"
        else:
            return f"Hi {user_name}! Welcome to Haqdarshak. I'm here to help you find government schemes and documents for {state_name} and central schemes. If you need support with digital skills, financial literacy, or growing your business, just ask. Let's work together to make your business successful!"
    
    else:
        if query_language == "Hindi":
            return f"नमस्ते {user_name}! हकदर्शक में स्वागत है। मैं यहाँ आपको {state_name} और केंद्रीय योजनाओं के लिए सरकारी योजनाएं और दस्तावेज़ खोजने में मदद करने के लिए हूँ।"
        else:
            return f"Hi {user_name}! Welcome to Haqdarshak. I'm here to help you find government schemes and documents for {state_name} and central schemes."
    # If user_type is 0, generate the original dynamic message
    prompt = f"""You are a helpful assistant for Haqdarshak, supporting small business owners in India with government schemes, digital/financial literacy, and business growth. The user is a new user named {user_name} from {state_name}.

    **Input**:
    - Query Language: {query_language}

    **Instructions**:
    - Generate a welcome message for a new user in the specified language ({query_language}).
    - For Hindi, use Devanagari script with simple, clear words suitable for micro business owners with low Hindi proficiency.
    - For English, use simple English with a friendly tone.
    - The message should welcome the user, and offer assistance with schemes and documents applicable to their state and all central government schemes or help with digital/financial literacy and business growth.
    - Response must be ≤70 words.
    - Start the response with 'Hi {user_name}!' (English) or 'नमस्ते {user_name}!' (Hindi).

    **Output**:
    - Return only the welcome message in the specified language.
    """

    try:
        response = llm.invoke([{"role": "user", "content": prompt}])
        generated_response = response.content.strip()
        logger.info(f"Generated welcome message in {query_language}: {generated_response}")
        return generated_response
    except Exception as e:
        logger.error(f"Failed to generate welcome message: {str(e)}")
        # Fallback to default messages
        if query_language == "Hindi":
            return f"नमस्ते {user_name}! हकदर्शक MSME चैटबॉट में स्वागत है। आप {state_name} से हैं, मैं आपकी राज्य और केंद्रीय योजनाओं में मदद करूँगा।"
        return f"Hi {user_name}! Welcome to Haqdarshak MSME Chatbot! Since you're from {state_name}, I'll help with schemes and documents applicable to your state and all central government schemes."

def generate_interaction_id(query, timestamp):
    return f"{query[:500]}_{timestamp.strftime('%Y%m%d%H%M%S')}"


# NEW: Async versions of your core functions
async def classify_intent_async(query: str, conversation_history: str = "") -> str:
    """Async version of classify_intent with caching"""
    # Check cache first
    cached_intent = await cache_manager.get_intent_cache(query, conversation_history)
    if cached_intent:
        logger.info(f"Intent cache hit: {cached_intent}")
        return cached_intent
    
    prompt = f"""You are an assistant for Haqdarshak. Classify the user's intent.

    **Input**:
    - Query: {query}
    - Conversation History: {conversation_history}

    **Instructions**:
    Return only one label from the following:
       - Schemes_Know_Intent - General queries enquiring about schemes or loans without specific names (e.g., 'show me schemes', 'mere liye schemes dikhao', 'loan', 'Schemes for credit?', 'MSME ke liye schemes kya hain?', 'क्रेडिट के लिए योजनाएं?', 'loan chahiye', 'scheme dikhao' etc.)
       - DFL_Intent - Digital/financial literacy queries (e.g., 'Current account', 'How to use UPI?', 'डिजिटल भुगतान कैसे करें?', 'Opening Bank Account', 'Why get Insurance', 'Why take loans', 'Online Safety', 'Setting up internet banking', 'Benefits of internet for business' etc.)
       - Specific_Scheme_Know_Intent - Queries that mention specific scheme names. Generally asking for loan or scheme is NOT specific. (e.g., 'What is FSSAI?', 'PMFME ke baare mein batao', 'एफएसएसएआई क्या है?', 'Pashu Kisan Credit Scheme ke baare mein bataiye', 'Udyam', 'Mudra Yojana', 'pmegp' , 'savitribai phule' , etc.)
       - Specific_Scheme_Apply_Intent - Queries about applying for specific schemes (e.g., 'Apply', 'Apply kaise karna hai', 'How to apply for FSSAI?', 'FSSAI kaise apply karu?', 'एफएसएसआईएआई के लिए आवेदन कैसे करें?' etc.)
       - Specific_Scheme_Eligibility_Intent - Queries about eligibility for specific schemes (e.g., 'Eligibility', 'Eligibility batao', 'Am I eligible for FSSAI?', 'FSSAI eligibility?', 'एफएसएसआईएआई की पात्रता क्या है?' etc.)
       - Out_of_Scope - Queries that are not relevant to business growth or digital literacy or financial literacy (e.g., 'What's the weather?', 'Namaste', 'मौसम कैसा है?', 'Time?' etc.)
       - Contextual_Follow_Up - Follow-up queries (e.g., 'Tell me more', 'Aur batao', 'और बताएं', 'iske baare mein aur jaankaari chahiye' etc.)
       - Confirmation_New_RAG - Confirmation for initiating another RAG search (Only to be chosen when user query is confirmation for initating another RAG search ("Yes", "Haan batao", "Haan dikhao", "Yes search again") AND previous assistant response says that the bot needs to fetch more details about some scheme. ('I need to fetch more details about [scheme name]. Please confirm if this is the scheme you meant.'))
       - Gratitude_Intent - User expresses thanks or acknowledgement (e.g., 'ok thanks', 'got it', 'theek hai', 'accha', 'thank you', 'शुक्रिया', 'धन्यवाद' etc.)

    **Tips**:
       - Use rule-based checks for Out_of_Scope (keywords: 'hello', 'hi', 'hey', 'weather', 'time', 'namaste', 'mausam', 'samay').
       - Single word queries with scheme names like 'pmegp', 'fssai', 'udyam' , 'savitribai phule' are in scope and should be classified as Specific_Scheme_Know_Intent.
       - For Contextual_Follow_Up, prioritise the most recent query-response pair from the conversation history to check if the query is a follow-up.
       - Use conversation history for context but intent should be determined solely by the current query.
       - To distinguish between Specific_Scheme_Know_Intent and Scheme_Know_Intent, check for whether query is asking for information about specific scheme or general information about schemes.
       - If some scheme name is mentioned in the query, then classify it as Specific_Scheme_Know_Intent.
    """
    
    try:
        response = await intent_llm.ainvoke([{"role": "user", "content": prompt}])
        intent = response.content.strip()
        
        # Cache the result
        await cache_manager.set_intent_cache(query, intent, conversation_history)
        return intent
    except Exception as e:
        logger.error(f"Failed to classify intent: {str(e)}")
        return "Out_of_Scope"

async def get_rag_response_async(query, vector_store, state="ALL_STATES", gender=None, business_category=None,userType=1):
    """Async version of get_rag_response"""
    try:
        details = []
        if state:
            details.append(f"state: {state}")
        if gender:
            details.append(f"gender: {gender}")
        if business_category:
            details.append(f"business category: {business_category}")

        full_query = query
        if details:
            full_query = f"{full_query}. {' '.join(details)}"

        # logger.debug(f"Processing query: {full_query}")
        
        # Run retrieval in thread pool since it's CPU intensive
        loop = asyncio.get_event_loop()
        
        def run_retrieval():
            
            retriever = PineconeRecordRetriever(
                index=vector_store, 
                state=state, 
                gender=gender,
                userType=userType, 
                k=5,
            )
            qa_chain = RetrievalQA.from_chain_type(
                llm=llm,
                chain_type="stuff",
                retriever=retriever,
                return_source_documents=True,
            )
            return qa_chain.invoke({"query": full_query})
        
        result = await loop.run_in_executor(executor, run_retrieval)
        response = result["result"]
        sources = result["source_documents"]
        
        logger.info(f"RAG response generated: {response}")
        return {"text": response, "sources": sources}
    except Exception as e:
        logger.error(f"RAG retrieval failed: {str(e)}")
        return {"text": "Error retrieving scheme information.", "sources": []}


async def get_scheme_response_async(
    query,
    vector_store,
    state="ALL_STATES",
    gender=None,
    business_category=None,
    include_mudra=False,
    intent=None,
    use_mongo=True,
    userType=1
):
    """Async version of get_scheme_response with MongoDB support"""
    logger.info("Querying scheme dataset")
    rag_start_time = time.perf_counter()   
    
    # Check cache first
    cache_context = {
        "state": state,
        "gender": gender,
        "business_category": business_category,
        "include_mudra": include_mudra,
        "intent": intent,
        "use_mongo": True
    }
    
    cached_response = await cache_manager.get_rag_cache(query, cache_context)
    if cached_response:
        logger.info("Scheme response cache hit")
        return cached_response

    guid = None
    rag = None
    
    # Regular Pinecone search
    logger.info(f"userType is equal too {userType}")
    rag = await get_rag_response_async(
                query,
                vector_store,
                state=state,
                gender=gender,
                business_category=business_category,
                userType=userType
            )

    if not isinstance(rag, dict):
        rag = {"text": str(rag), "sources": []}

    # rag["text"] = f"{rag.get('text', '')}\n{mudra_rag.get('text', '')}"
    # rag["sources"] = rag.get("sources", []) + mudra_rag.get("sources", [])

    total_rag_time = time.perf_counter() - rag_start_time
    logger.info(f"Total RAG processing time: {total_rag_time:.3f}s")    

    # Cache the result
    await cache_manager.set_rag_cache(query, rag, cache_context)
    
    return rag

async def get_dfl_response_async(query, vector_store, state=None, gender=None, business_category=None):
    """Async wrapper for DFL dataset retrieval"""
    logger.info("Querying DFL dataset")
    return await get_rag_response_async(
        query,
        vector_store,
        state=None,
        gender=gender,
        business_category=business_category,
    )

def get_language_config(language):
    """Get language-specific configuration"""
    # Handle language code to name mapping
    if language in LANGUAGE_LABELS:
        language_name = LANGUAGE_LABELS[language]
    else:
        if language == "English":
            language_name = "Hindi"
        language_name = language
    
    # Return config for the language, fallback to English if not found

    return LANGUAGE_CONFIG.get(language_name, LANGUAGE_CONFIG["Hindi"]), language_name

async def generate_response_async(
    intent: str, 
    rag_response: str, 
    user_info: UserContext, 
    language: str, 
    context: str, 
    query: str, 
    scheme_guid: str = None, 
    stream: bool = False
):
    """Updated async version with proper multilingual support"""
    print(f"Generating response for intent: {intent}, language: {language}, query: {query} and rag_response: {rag_response}...")
    
    # Get language configuration
    lang_config, language_name = get_language_config(language)
    print(f"Language :{lang_config} detected: {language_name}")
    print(f"Using language config for: {language_name} and {lang_config}")
    # Handle out of scope
    if intent == "Out_of_Scope":
        response = lang_config["out_of_scope"]
        if stream:
            async def stream_response():
                for char in response:
                    yield char
            return stream_response()
        return response

    # Handle gratitude
    if intent == "Gratitude_Intent":
        gratitude_prompt = f"""You are a friendly assistant for Haqdarshak. The user {user_info.name} has thanked you.

        **Instructions**:
        - Respond briefly in {language_name} acknowledging the thanks and offering further help.
        - Keep the message under 30 words.
        - Use natural, conversational tone appropriate for {language_name}.

        **Output**:
        - Only the acknowledgement message in {language_name}."""
        
        try:
            if stream:
                async def stream_gratitude():
                    async for chunk in llm.astream([{"role": "user", "content": gratitude_prompt}]):
                        token = chunk.content or ""
                        if token:
                            yield token
                return stream_gratitude()
            else:
                response = await llm.ainvoke([{"role": "user", "content": gratitude_prompt}])
                return response.content.strip()
        except Exception as e:
            logger.error(f"Failed to generate gratitude response: {str(e)}")
            fallback_response = lang_config["gratitude_fallback"]
            
            if stream:
                async def stream_fallback():
                    for char in fallback_response:
                        yield char
                return stream_fallback()
            return fallback_response

    # Build the main prompt for other intents
    word_limit = 150 if intent == "Schemes_Know_Intent" else 100
    tone_prompt = get_system_prompt(language_name, user_info.name, word_limit)

    greeting_text = lang_config["greeting"]

    base_prompt = f"""You are a helpful assistant for Haqdarshak, supporting small business owners in India with government schemes, digital/financial literacy, and business growth.

    **Input**:
    - Intent: {intent}
    - RAG Response: {rag_response}
    - Current Query: {query}
    - User Name: {user_info.name}
    - State: {user_info.state_name} ({user_info.state_id})
    - Gender: {user_info.gender}
    - Business Name: {user_info.business_name}
    - Business Category: {user_info.business_category}
    - Conversation Context: {context}
    - Language: {language_name}"""
    
    if scheme_guid:
        base_prompt += f"\n    - Scheme GUID: {scheme_guid}"

    base_prompt += f"""

    **Language Instructions**:
    - Respond ONLY in {language_name}
    - Use natural, conversational tone appropriate for {language_name}
    - Maintain cultural context and appropriate formality level
    {tone_prompt}

    **Formatting Instructions**:
    - Start with greeting: '{greeting_text} {user_info.name}!'
    - After greeting, add a blank line
    - Structure answer in multiple short paragraphs (1-2 lines each)
    - Add blank line between paragraphs for readability
    - Use clear, simple formatting with bullet points when appropriate

    **Task**:
    Use user-provided scheme details to pick relevant schemes from retrieved data and personalise information.
    Prioritise the **Current Query** over **Conversation Context**.
    """

    special_schemes = ["Udyam", "FSSAI", "Shop Act", "GST", "Mudra", "PMEGP", "PMFME", "CMEGP", "Yuva Udyami", "PMSBY", "PMJJBY", "PMJAY (Ayushman Bharat)"]

    # Build intent-specific prompts using language config
    if intent == "Specific_Scheme_Know_Intent":
        intent_prompt = (
            "Share scheme name, purpose, benefits and other fetched relevant details in structured format from **RAG Response**. "
            f"Ask: '{lang_config['eligibility_question']}'"
        )
        intent_prompt += (
            f" For {', '.join(special_schemes)}, add: '{lang_config['haqdarshak_message']}'"
        )
            
    elif intent == "Specific_Scheme_Apply_Intent":
        intent_prompt = (
            "Share application process from **RAG Response**."
        )
        intent_prompt += (
            f" For {', '.join(special_schemes)}, add: '{lang_config['haqdarshak_message']}'"
        )
            
    elif intent == "Specific_Scheme_Eligibility_Intent":
        intent_prompt = (
            "Summarize eligibility rules from **RAG Response** and provide a link "
            f"to check eligibility: https://customer.haqdarshak.com/check-eligibility/{scheme_guid}. "
            "Ask the user to verify their eligibility there."
        )
        intent_prompt += (
            f" For {', '.join(special_schemes)}, add: '{lang_config['haqdarshak_message']}'"
        )
            
    elif intent == "Schemes_Know_Intent":
        intent_prompt = (
            "List 3-4 schemes from **RAG Response** with short one-line description for each. "
            "Always include Pradhan Mantri Mudra Yojana as one of the schemes. "
            "Use user provided scheme details to choose most relevant schemes. "
            "If no close match found, list top schemes applicable to user in their state or CSS. "
            f"Finally ask: '{lang_config['scheme_question']}'"
        )
        intent_prompt += (
            f" For {', '.join(special_schemes)}, add: '{lang_config['haqdarshak_message']}' "
            "Add this only in description for applicable scheme/s, not under entire list."
        )
            
    elif intent == "DFL_Intent":
        intent_prompt = (
            f"Use **RAG Response** if available, augmenting with your knowledge where relevant. "
            f"If RAG Response is empty or not relevant, provide helpful answer from your knowledge "
            f"in simple {language_name} with helpful examples."
        )
        
    elif intent == "Contextual_Follow_Up":
        intent_prompt = (
            "Use Previous Assistant Response and Conversation Context to identify topic. "
            "If RAG Response doesn't match referenced scheme, indicate new RAG search needed. "
            "Provide relevant follow-up response using RAG Response, filtering for schemes where "
            "'applicability' includes state_id or 'scheme type' is 'Centrally Sponsored Scheme' (CSS). "
            f"If unclear, ask for clarification (e.g., '{lang_config['which_scheme']}')"
        )
            
    elif intent == "Confirmation_New_RAG":
        intent_prompt = (
            "If user confirms to initiate new RAG search, respond with details of "
            "scheme they are interested in, refer to conversation context for details."
        )
    else:
        intent_prompt = ""

    output_prompt = f"""
    **Output**:
    - Return only the final response in {language_name} (no intent label or intermediate steps)
    - If new RAG search needed for schemes, indicate with: '{lang_config['rag_search']}'
    - If RAG Response is empty or 'No relevant scheme information found,' and query is Contextual_Follow_Up referring to specific scheme, indicate new RAG search needed. Otherwise, say: '{lang_config['no_info']}'
    - Do not mention other schemes when specific scheme is being discussed
    - When intent is Schemes_Know, only mention current relevant schemes, not past conversation schemes
    - Include user profile details only where contextually relevant
    - Scheme answers must come only from scheme data. For DFL answers, use DFL document supplemented by your knowledge when possible
    """

    prompt = f"{base_prompt}{intent_prompt}\n{output_prompt}"

    try:
        if stream:
            async def stream_main_response():
                buffer = ""
                try:
                    async for chunk in llm.astream([{"role": "user", "content": prompt}]):
                        token = chunk.content or ""
                        buffer += token
                        if token:
                            yield token
                
                    # Add eligibility link for specific intent after streaming
                    if intent == "Specific_Scheme_Eligibility_Intent" and scheme_guid:
                        screening_link = f"https://customer.haqdarshak.com/check-eligibility/{scheme_guid}"
                        if screening_link not in buffer:
                            link_text = f"\n{screening_link}"
                            for char in link_text:
                                yield char
                            
                except Exception as e:
                    logger.error(f"Failed to stream response: {str(e)}")
                    error_msg = lang_config["error_message"]
                    for char in error_msg:
                        yield char
            
            return stream_main_response()
        else:
            response = await llm.ainvoke([{"role": "user", "content": prompt}])
            final_text = response.content.strip()
            
            if intent == "Specific_Scheme_Eligibility_Intent" and scheme_guid:
                screening_link = f"https://customer.haqdarshak.com/check-eligibility/{scheme_guid}"
                if screening_link not in final_text:
                    final_text += f"\n{screening_link}"
            
            return final_text
    except Exception as e:
        logger.error(f"Failed to generate response: {str(e)}")
        error_response = lang_config["error_message"]
        
        if stream:
            async def stream_error():
                for char in error_response:
                    yield char
            return stream_error()
        return error_response


def generate_hindi_audio_script(
    original_response: str,
    user_info: UserContext,
    rag_response: str = "",
) -> str:
    """
    Generates a summarized, human-like Hindi script for text-to-speech from the original bot response.
    The script should avoid punctuation marks and focus on natural flow.
    """
    prompt = f"""You are an assistant for Haqdarshak. Your task is to summarize the provided text into a concise, human-like script
    in natural Hindi (Devanagari script) for a text-to-speech system.
    
    **Instructions**:
    - Summarize the core information from the provided 'Final Response' and 'RAG Response'.
    - Ensure the summary flows naturally as if spoken by a human.
    - Translate the summary into clear and simple Hindi (Devanagari script) using simple hindi words.
    - Focus on the main points and keep the summary concise, between 50-100 words, to ensure a smooth audio experience.
    - The response should be purely the Hindi script, with no introductory or concluding remarks.
    - For number ranges like "10%-20%", use "10 se 20" in Hindi.
    - Do NOT use any english words. 
    - Do NOT translate Smileys or emoticons.
    - Always use simpler alternatives wherever the words are in complex hindi e.g. Instead of "vyavyasay" say "business", instead of "vanijya" say "finance"
    - Do NOT include urls and web links. 

    **Final Response**:
    {original_response}

    **RAG Response**:
    {rag_response}

    **Output**:
    """
    try:
        response = llm.invoke([{"role": "user", "content": prompt}])
        hindi_script = response.content.strip()
        logger.info(f"Generated Hindi audio script: {hindi_script}")
        return hindi_script
    except Exception as e:
        logger.error(f"Failed to generate Hindi audio script: {str(e)}")
        try:
            translation_prompt = f"Translate the following text into simple Hindi (Devanagari script), removing all punctuation and hyphens for a smooth audio output: {original_response}"
            translation_response = llm.invoke([{"role": "user", "content": translation_prompt}])
            hindi_script = translation_response.content.strip()
            logger.warning(f"Falling back to direct translation for Hindi audio script: {hindi_script}")
            return hindi_script
        except Exception as inner_e:
            logger.error(f"Failed to fall back to direct translation: {str(inner_e)}")
            return "ऑडियो स्क्रिप्ट उत्पन्न करने में त्रुटि हुई है।"

# Background task functions
async def save_conversation_background(
    session_id: str, 
    mobile_number: str, 
    query: str, 
    response: str,
    hindi_script: str = ""
):
    """Save conversation in background without blocking response"""
    try:
        interaction_id = generate_interaction_id(query, datetime.utcnow())
        messages_to_save = [
            {"role": "user", "content": query, "timestamp": datetime.utcnow(), "interaction_id": interaction_id},
            {"role": "assistant", "content": response, "timestamp": datetime.utcnow(), "interaction_id": interaction_id, "audio_script": hindi_script},
        ]
        
        await data_manager.save_conversation_async(session_id, mobile_number, messages_to_save)
        logger.info(f"Background save completed for session {session_id} (Interaction ID: {interaction_id})")
    except Exception as e:
        logger.error(f"Background save failed for session {session_id}: {str(e)}")

async def generate_audio_script_background(response: str, user_info: UserContext, rag_response: str = "") -> str:
    """Generate hindi audio script in background"""
    try:
        loop = asyncio.get_event_loop()
        hindi_script = await loop.run_in_executor(
            executor, 
            generate_hindi_audio_script,
            response,
            user_info,
            rag_response
        )
        return hindi_script
    except Exception as e:
        logger.error(f"Background audio script generation failed: {str(e)}")
        return "ऑडियो स्क्रिप्ट उत्पन्न करने में त्रुटि हुई है।"

async def get_popular_scheme_response_fast(query: str, intent: str,userType: int) -> Optional[dict]:
    """Ultra-fast response for popular schemes using MongoDB (1-2 seconds)"""

    logger.info(f"Processing popular userType {userType} scheme query: {query} with intent: {intent}")
    # if intent != "Specific_Scheme_Know_Intent":
    #     return None
    
    # Step 1: Quick GUID lookup from MongoDB (< 100ms)
    loop = asyncio.get_event_loop()
    guid = await loop.run_in_executor(executor, find_scheme_guid_by_query, query,userType)
    
    if not guid:
        logger.info(f"No popular scheme GUID found for query: '{query}' - using regular path")
        return None
    
    logger.info(f"Found popular scheme GUID: {guid} for query: '{query}' - using fast path")
    
    # Step 2: Fast MongoDB fetch (< 200ms)  
    if not MONGO_SCHEME_AVAILABLE:
        logger.warning("MongoDB not available - falling back to regular path")
        return None
    
    try:
        # Single fast operation - fetch docs from MongoDB
        logger.info("starting scheme fetch by guid")
        docs = await loop.run_in_executor(
            executor, 
            fetch_scheme_docs_by_guid, 
            guid, 
            None,
            True,
            userType,
        )
        if not docs:
            logger.warning(f"No docs found for {docs} popular scheme GUID: {guid}")
            return None
        
        # Step 3: Fast QA chain (< 1000ms)
        def run_fast_qa():
            retriever = DocumentListRetriever(docs)
            qa_chain = RetrievalQA.from_chain_type(
                llm=llm,
                chain_type="stuff", 
                retriever=retriever,
                return_source_documents=True,
                
            )
            return qa_chain.invoke({"query": query})
        
        result = await loop.run_in_executor(executor, run_fast_qa)
        rag_response = {"text": result["result"], "sources": result["source_documents"]}
        logger.info(f"Fast path completed for popular scheme: {guid}")
        return rag_response
        
    except Exception as e:
        logger.error(f"Fast path failed for GUID {guid}: {str(e)} - falling back")
        return None

def create_audio_task_background(response: str, user_info: UserContext, rag_response: str = ""):
    """Create a background audio task that returns a coroutine"""
    async def audio_task(final_text: str = None) -> str:
        text_to_use = final_text or response
        return await generate_audio_script_background(text_to_use, user_info, rag_response)
    return audio_task

# Performance tracking context manager
class PerformanceTracker:
    def __init__(self):
        self.timings = {}
        self.start_time = time.perf_counter()
    
    def start_timer(self, operation_name: str):
        self.timings[f"{operation_name}_start"] = time.perf_counter()
    
    def end_timer(self, operation_name: str):
        start_key = f"{operation_name}_start"
        if start_key in self.timings:
            elapsed = time.perf_counter() - self.timings[start_key]
            self.timings[operation_name] = elapsed
            del self.timings[start_key]
    
    def log_summary(self):
        total = time.perf_counter() - self.start_time
        operations = {k: v for k, v in self.timings.items() if not k.endswith('_start')}
        summary = "\n".join(f"{k}: {v:.3f}s" for k, v in operations.items())
        summary += f"\nTotal: {total:.3f}s"
        logger.info(f"Performance summary:\n{summary}")

# MAIN OPTIMIZED ASYNC FUNCTION
async def process_query_optimized(
    query: str,
    scheme_vector_store,
    dfl_vector_store,
    session_id: str,
    mobile_number: str,
    session_data: SessionData,
    userType: int = 1,
    user_language: str = None,
    stream: bool = False
) -> Tuple[Any, callable]:
    """
    Optimized async version of process_query with parallel processing and caching
    Expected improvement: 12.81s -> 3-4s (70% improvement)
    """
    tracker = PerformanceTracker()
    logger.info(f"Starting optimized userType {userType} query processing for: {query}")

    # Step 1: Get user context (fast, local operation)
    tracker.start_timer("user_context")
    user_info = get_user_context(session_data)
    tracker.end_timer("user_context")
    
    if not user_info:
        tracker.log_summary()
        return "Error: User not logged in.", None

    # Step 2: Language detection (fast, local operation)
    tracker.start_timer("language_detection")
    query_language = user_language
    #  if query.lower() == "welcome" and user_language else detect_language(query)
    tracker.end_timer("language_detection")
    
    logger.info(f"Using query language: {query_language}")

    # Step 3: Start background conversation fetch early
    tracker.start_timer("fetch_conversations")
    conversations_task = asyncio.create_task(data_manager.get_conversations_async(mobile_number))

    # Step 4: Handle welcome query (early return)
    if query.lower() == "welcome":
        conversations = await conversations_task
        tracker.end_timer("fetch_conversations")
        user_type = "returning" if conversations else "new"
        
        if user_type == "new":
            response = get_welcome_message(user_info.state_name, user_info.name, query_language,userType)
            
            # Create background task for saving welcome message
            async def save_welcome():
                try:
                    interaction_id = generate_interaction_id(response, datetime.utcnow())
                    messages = [{"role": "assistant", "content": response, "timestamp": datetime.utcnow(), "interaction_id": interaction_id}]
                    await data_manager.save_conversation_async(session_id, mobile_number, messages)
                    logger.info(f"Saved welcome message for new user in session {session_id}")
                except Exception as e:
                    logger.error(f"Failed to save welcome message: {str(e)}")
            
            # Start background save but don't wait
            asyncio.create_task(save_welcome())
            
            audio_task = create_audio_task_background(response, user_info)
            tracker.log_summary()
            
            if stream:
                async def gen():
                    for ch in response:
                        yield ch
                return gen(), audio_task
            return response, audio_task
        else:
            tracker.log_summary()
            return None, None

    # Step 5: Build conversation history (fast, local)
    conversation_history = build_conversation_history(session_data.messages)
    
    # Step 6: Start intent classification early (parallel with conversation fetch)
    tracker.start_timer("intent_classification")
    intent_task = asyncio.create_task(classify_intent_async(query, conversation_history))
    # intent_task = asyncio.create_task(classify_intent_rules_only(query, conversation_history))
    
    # Step 7: Wait for conversations and get user type
    conversations = await conversations_task
    tracker.end_timer("fetch_conversations")
    user_type = "returning" if conversations else "new"

    # Step 8: Get intent result
    intent = await intent_task
    tracker.end_timer("intent_classification")
    logger.info(f"Classified intent: {intent}")

    # Step 9: Determine context and prepare for RAG
    follow_up_intents = {
        "Contextual_Follow_Up",
        "Specific_Scheme_Eligibility_Intent", 
        "Specific_Scheme_Apply_Intent",
        "Confirmation_New_RAG",
    }
    follow_up = intent in follow_up_intents
    
    # Get recent conversation context only for follow-ups
    recent_query = None
    recent_response = None
    if follow_up and session_data.messages:
        for msg in reversed(session_data.messages):
            if msg["role"] == "assistant" and "Welcome" not in msg["content"]:
                recent_response = msg["content"]
                msg_index = session_data.messages.index(msg)
                if msg_index > 0 and session_data.messages[msg_index - 1]["role"] == "user":
                    recent_query = session_data.messages[msg_index - 1]["content"]
                break
    
    context_pair = f"User: {recent_query}\nAssistant: {recent_response}" if follow_up and recent_query and recent_response else ""
    


    scheme_intents = {"Specific_Scheme_Know_Intent", "Specific_Scheme_Apply_Intent", "Specific_Scheme_Eligibility_Intent", "Schemes_Know_Intent", "Contextual_Follow_Up", "Confirmation_New_RAG"}
    dfl_intents = {"DFL_Intent", "Non_Scheme_Know_Intent"}

    logger.info(f"Processing query: '{query}' with intent: {intent}")

    rag_response = None
    if intent in scheme_intents:
        tracker.start_timer("rag_retrieval")
        logger.info(f"Retrieving RAG response for kittu intent: {intent}")
        # TRY FAST PATH FIRST for popular schemes (1-2 seconds)
        if intent == "Specific_Scheme_Know_Intent" or "Schemes_Know_Intent":
            rag_response = await get_popular_scheme_response_fast(query, intent,userType)
            logger.info(f"Fast path response: {rag_response}")
        # FALLBACK to full pipeline if fast path didn't work
        if not rag_response:
            logger.info("Using full scheme response pipeline")
            include_mudra = intent == "Schemes_Know_Intent"
            
            rag_response = await get_scheme_response_async(
                query=query,
                vector_store=scheme_vector_store,
                state=user_info.state_id,
                gender=user_info.gender,
                business_category=user_info.business_category,
                include_mudra=include_mudra,
                intent=intent,
                use_mongo=True,
                userType=userType
            )
        
        tracker.end_timer("rag_retrieval")

    elif intent in dfl_intents:
        tracker.start_timer("dfl_retrieval")
        
        rag_response = await get_dfl_response_async(
            query=query,
            vector_store=dfl_vector_store,
            state=user_info.state_id,
            gender=user_info.gender,
            business_category=user_info.business_category
        )
        
        tracker.end_timer("dfl_retrieval")

    # Step 11: Generate response (this is where the fix is important)
    tracker.start_timer("generate_response")
    rag_text = rag_response.get("text") if isinstance(rag_response, dict) else rag_response
    if intent == "DFL_Intent" and (rag_text is None or "No relevant" in rag_text):
        rag_text = ""
    scheme_guid = None
    if isinstance(rag_response, dict) and intent == "Specific_Scheme_Eligibility_Intent":
        scheme_guid = extract_scheme_guid(rag_response.get("sources", []))

    response_result = await generate_response_async(
        intent,
        rag_text or "",
        user_info,
        query_language,
        context_pair,
        query,
        scheme_guid=scheme_guid,
        stream=stream,
    )
    tracker.end_timer("generate_response")

    # Step 12: Handle streaming vs non-streaming audio tasks
    if stream:
        # For streaming, response_result is an async generator
        def create_streaming_audio_task():
            async def streaming_audio_task(final_text: str) -> str:
                try:
                    hindi_script = await generate_audio_script_background(final_text, user_info, rag_text or "")
                    # Fire and forget the save task
                    asyncio.create_task(
                        save_conversation_background(session_id, mobile_number, query, final_text, hindi_script)
                    )
                    return hindi_script
                except Exception as e:
                    logger.error(f"Audio script generation failed: {e}")
                    return "ऑडियो स्क्रिप्ट उत्पन्न करने में त्रुटि हुई है।"
            
            return streaming_audio_task
    
        audio_task = create_streaming_audio_task()
    else:
        # For non-streaming, response_result is a string
        response_text = response_result
        
        async def background_audio_task(final_text: str = None) -> str:
            text_to_use = final_text or response_text
            
            try:
                hindi_script = await generate_audio_script_background(text_to_use, user_info, rag_text or "")
            except Exception as e:
                logger.error(f"Audio script generation failed: {e}")
                hindi_script = "ऑडियो स्क्रिप्ट उत्पन्न करने में त्रुटि हुई है।"
            
            # Start save task in fire-and-forget mode
            asyncio.create_task(
                save_conversation_background(session_id, mobile_number, query, text_to_use, hindi_script)
            )
            
            return hindi_script
        
        audio_task = background_audio_task

    tracker.log_summary()
    logger.info(f"Query processing completed for: {query}")

    return response_result, audio_task


def run_qa_chain_fast(docs, query):
    """Fast QA chain execution"""
    retriever = DocumentListRetriever(docs)
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
    )
    result = qa_chain.invoke({"query": query})
    return {"text": result["result"], "sources": result["source_documents"]}