-- Calibrus — optional MariaDB schema for users who want a dedicated table.
-- Usage: mysql -u root -p trading < scripts/setup_db.sql

CREATE DATABASE IF NOT EXISTS trading
  CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;

USE trading;

CREATE TABLE IF NOT EXISTS signals (
    id          BIGINT UNSIGNED AUTO_INCREMENT PRIMARY KEY,
    timestamp   DATETIME NOT NULL,
    open        DECIMAL(18,8) NOT NULL,
    high        DECIMAL(18,8) NOT NULL,
    low         DECIMAL(18,8) NOT NULL,
    close       DECIMAL(18,8) NOT NULL,
    volume      DECIMAL(24,8) NOT NULL,
    text        TEXT NULL,
    label       ENUM('short', 'hold', 'long') NULL,
    created_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

    INDEX idx_timestamp (timestamp)
) ENGINE=InnoDB;

-- Dedicated application user (recommended instead of using root)
-- Replace 'CHANGE_ME' with a strong password, then export it as the
-- CALIBRUS_DB_PASSWORD environment variable rather than putting it in config.yaml.
CREATE USER IF NOT EXISTS 'calibrus_user'@'localhost' IDENTIFIED BY 'CHANGE_ME';
GRANT SELECT, INSERT ON trading.signals TO 'calibrus_user'@'localhost';
FLUSH PRIVILEGES;
