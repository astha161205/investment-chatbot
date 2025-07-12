<?php
header('Content-Type: application/json');
header('Access-Control-Allow-Origin: *');
header('Access-Control-Allow-Methods: GET');

// Database configuration
$host = 'localhost';
$dbname = 'investment_chatbot'; // Replace with your database name

$dbusername="root";
$dbpassword = "";

try {
    // Create database connection
    $pdo = new PDO("mysql:host=$host;dbname=$dbname;charset=utf8", $dbusername, $dbpassword);
    $pdo->setAttribute(PDO::ATTR_ERRMODE, PDO::ERRMODE_EXCEPTION);
    
    // Get approved testimonials, ordered by creation date (newest first)
    $sql = "SELECT name, role, rating, testimonial, created_at FROM testimonials WHERE status = 'approved' ORDER BY created_at DESC LIMIT 6";
    $stmt = $pdo->prepare($sql);
    $stmt->execute();
    
    // Fetch all testimonials
    $testimonials = $stmt->fetchAll(PDO::FETCH_ASSOC);
    
    // Send testimonials as JSON
    echo json_encode($testimonials);
    
} catch (PDOException $e) {
    // Database error
    error_log("Database Error: " . $e->getMessage());
    echo json_encode([]);
} catch (Exception $e) {
    // Other error
    error_log("Error: " . $e->getMessage());
    echo json_encode([]);
}
?> 