<?php
// Enable error reporting for debugging
error_reporting(E_ALL);
ini_set('display_errors', 1);

// Database configuration
$host = 'localhost';
$dbname = 'investment_chatbot';
$dbusername = "root";
$dbpassword = "";

try {
    // Create database connection
    $pdo = new PDO("mysql:host=$host;dbname=$dbname;charset=utf8", $dbusername, $dbpassword);
    $pdo->setAttribute(PDO::ATTR_ERRMODE, PDO::ERRMODE_EXCEPTION);
    
    echo "✅ Database connection successful!<br>";
    
    // Check if testimonials table exists
    $stmt = $pdo->query("SHOW TABLES LIKE 'testimonials'");
    if ($stmt->rowCount() > 0) {
        echo "✅ Testimonials table exists!<br>";
        
        // Check table structure
        $stmt = $pdo->query("DESCRIBE testimonials");
        echo "📋 Table structure:<br>";
        while ($row = $stmt->fetch(PDO::FETCH_ASSOC)) {
            echo "- {$row['Field']}: {$row['Type']} ({$row['Null']})<br>";
        }
        
        // Check if table is empty
        $stmt = $pdo->query("SELECT COUNT(*) as count FROM testimonials");
        $count = $stmt->fetch(PDO::FETCH_ASSOC)['count'];
        echo "📊 Current testimonials count: $count<br>";
        
    } else {
        echo "❌ Testimonials table does not exist!<br>";
    }
    
} catch (PDOException $e) {
    echo "❌ Database Error: " . $e->getMessage() . "<br>";
} catch (Exception $e) {
    echo "❌ Error: " . $e->getMessage() . "<br>";
}
?> 