import os
from dotenv import load_dotenv
from supabase import create_client
load_dotenv()
supabase = create_client(os.getenv('SUPABASE_URL'), os.getenv('SUPABASE_KEY'))
res = supabase.table('travel_chunks').select('city').limit(5).execute()
print(res.data)
