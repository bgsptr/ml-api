CREATE TYPE user_wallet_information AS (
    user_email VARCHAR,
    user_fname VARCHAR,
    user_lname VARCHAR,
    wallet_id VARCHAR,
    wallet_balance INT8, --supabase int8 same with bingint in sql
    wallet_created TIMESTAMP,
    wallet_status VARCHAR
);

DROP TYPE IF EXISTS user_wallet_information;

CREATE OR REPLACE FUNCTION get_user_wallet_information(email_user VARCHAR)
RETURNS SETOF user_wallet_information
LANGUAGE sql
AS $$
    SELECT 
        users.email AS user_email, 
        users.fname AS user_fname, 
        users.lname AS user_lname, 
        wallets.id_wallet AS wallet_id, 
        wallets.balance AS wallet_balance,
        wallets.created_at AS wallet_created, 
        wallets.status AS wallet_status 
    FROM 
        users 
    LEFT JOIN 
        wallets ON users.email = wallets.email
    WHERE
        users.email = email_user;
$$;

CREATE OR REPLACE FUNCTION get_user_specific_wallet_information(email_user VARCHAR, wallet_id_param VARCHAR)
RETURNS SETOF user_wallet_information
LANGUAGE sql
AS $$
    SELECT 
        users.email AS user_email, 
        users.fname AS user_fname, 
        users.lname AS user_lname, 
        wallets.id_wallet AS wallet_id, 
        wallets.balance AS wallet_balance,
        wallets.created_at AS wallet_created, 
        wallets.status AS wallet_status 
    FROM 
        users 
    LEFT JOIN 
        wallets ON users.email = wallets.email
    WHERE
        users.email = email_user
    AND
        wallets.id_wallet = wallet_id_param;
$$;


-- create or replace function get_user_wallet_information()
-- returns setof user_wallet_information
-- language sql
-- as $$
--     select users.email AS user_email, users.fname AS user_fname, users.lname AS user_lname, wallet.id_wallet AS wallet_id, wallets.balance AS wallet_balance, wallet.created_at AS wallet_created, wallet.status AS wallet_status from users left join wallets on users.email = wallets.email;
-- $$;