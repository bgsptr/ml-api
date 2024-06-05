from supabase import create_client, Client
import os

# supabase_url = os.environ.get('REACT_APP_SUPABASE_URL')
# supabase_key = os.environ.get('REACT_APP_SUPABASE_ANON_KEY')

supabase_url = "https://kkhalvpbjejearwzoutq.supabase.co"
supabase_key = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImtraGFsdnBiamVqZWFyd3pvdXRxIiwicm9sZSI6ImFub24iLCJpYXQiOjE3MTcyMDkwNjIsImV4cCI6MjAzMjc4NTA2Mn0.1FIgeZ5j7HWHUExleoSbkg7fXQR1VWy0hvGuOaPPJTQ"

supabase_client: Client = create_client(supabase_url, supabase_key)

def get_supabase_client() -> Client:
    return supabase_client