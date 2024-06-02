CREATE TABLE users(
  email varchar(255) unique not null,
  password varchar(255) not null,
  fname varchar(255) not null,
  lname varchar(255) not null,
  created_at timestamp,
  PRIMARY KEY (email)
)

CREATE TABLE bottles(
    id_bottle int(20) unique not null,
    bottle_name varchar(255) unique not null,
    price int(100) not null,
    PRIMARY KEY (id_bottle)
)

CREATE TABLE transactions(
    id_transaction int(20) unique not null,
    email varchar(255) not null,
    created_at timestamp,
    price_total int(100) not null,
    PRIMARY KEY (id_transaction),
    FOREIGN KEY (email) references users(email)
)

CREATE TABLE transactions_bottles(
    id_tb int(20) unique not null,
    id_bottle int(20) not null,
    id_transaction int(20) not null,
    file_path varchar(255) not null,
    PRIMARY KEY (id_tb),
    FOREIGN KEY (id_bottle) references bottles(id_bottle),
    FOREIGN KEY (id_transaction) references transactions(id_transaction)
)

CREATE TABLE roles(
    id_role int(20) unique not null,
    name_role varchar(255) unique not null,
    PRIMARY KEY (id_role)
)

CREATE TABLE users_roles(
    id_role int(20) not null,
    email varchar(255) not null,
    PRIMARY KEY (id_role, email),
    FOREIGN KEY (id_role) REFERENCES roles(id_role),
    FOREIGN KEY (email) REFERENCES users(email)
)

CREATE TABLE wallets(
    id_wallet int(20) unique not null,
    balance int(100) not null,
    email varchar(255) not null,
    PRIMARY KEY (id_wallet)
    FOREIGN KEY (email) REFERENCES users(email)
)