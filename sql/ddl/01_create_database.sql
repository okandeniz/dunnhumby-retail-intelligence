-- Create the project database
CREATE DATABASE IF NOT EXISTS dunnhumby_retail
CHARACTER SET utf8mb4
COLLATE utf8mb4_unicode_ci;

-- Verify database creation
SHOW DATABASES LIKE 'dunnhumby_retail';

-- Select the database
USE dunnhumby_retail;

-- Check the database
SELECT DATABASE() AS current_database;