-- Calibrus — schéma MariaDB optionnel pour les users voulant une table dédiée.
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

-- Utilisateur applicatif dédié (recommandé plutôt que d'utiliser root)
-- Remplace 'CHANGE_ME' par un mot de passe fort, puis exporte-le en variable
-- d'environnement CALIBRUS_DB_PASSWORD plutôt que de le mettre dans config.yaml.
CREATE USER IF NOT EXISTS 'calibrus_user'@'localhost' IDENTIFIED BY 'CHANGE_ME';
GRANT SELECT, INSERT ON trading.signals TO 'calibrus_user'@'localhost';
FLUSH PRIVILEGES;
