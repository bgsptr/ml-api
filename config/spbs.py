from supabase import create_client, Client
import os

supabase_url = os.getenv('REACT_APP_SUPABASE_URL')
supabase_key = os.getenv('REACT_APP_SUPABASE_ANON_KEY')

supabase_client: Client = create_client(supabase_url, supabase_key)

def get_supabase_client() -> Client:
    return supabase_client