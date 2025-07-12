<?php
header('Content-Type: application/json');
header('Access-Control-Allow-Origin: *');
header('Access-Control-Allow-Methods: POST');
header('Access-Control-Allow-Headers: Content-Type');

// Enable error reporting for debugging
error_reporting(E_ALL);
ini_set('display_errors', 1);

// Database configuration
$host = 'localhost';
$dbname = 'investment_chatbot';

$dbusername="root";
$dbpassword = "";

try {
    // Create database connection
    $pdo = new PDO("mysql:host=$host;dbname=$dbname;charset=utf8", $dbusername, $dbpassword);
    $pdo->setAttribute(PDO::ATTR_ERRMODE, PDO::ERRMODE_EXCEPTION);
    
    // Check if it's a POST request
    if ($_SERVER['REQUEST_METHOD'] !== 'POST') {
        throw new Exception('Invalid request method');
    }
    
    // Debug: Log the POST data
    error_log("POST data received: " . print_r($_POST, true));
    
    // Get form data
    $name = trim($_POST['name'] ?? '');
    $role = trim($_POST['role'] ?? '');
    $email = trim($_POST['email'] ?? '');
    $rating = intval($_POST['rating'] ?? 0);
    $testimonial = trim($_POST['testimonial'] ?? '');
    
    // Debug: Log the processed data
    error_log("Processed data - Name: $name, Role: $role, Email: $email, Rating: $rating, Testimonial: $testimonial");
    
    // Validate required fields
    if (empty($name)) {
        throw new Exception('Name is required');
    }
    
    if (empty($testimonial)) {
        throw new Exception('Testimonial is required');
    }
    
    if ($rating < 1 || $rating > 5) {
        throw new Exception('Rating must be between 1 and 5');
    }
    
    // Validate email if provided
    if (!empty($email) && !filter_var($email, FILTER_VALIDATE_EMAIL)) {
        throw new Exception('Invalid email format');
    }
    
    // Sanitize inputs
    $name = htmlspecialchars($name, ENT_QUOTES, 'UTF-8');
    $role = htmlspecialchars($role, ENT_QUOTES, 'UTF-8');
    $email = htmlspecialchars($email, ENT_QUOTES, 'UTF-8');
    $testimonial = htmlspecialchars($testimonial, ENT_QUOTES, 'UTF-8');
    
    // Prepare SQL statement
    $sql = "INSERT INTO testimonials (name, role, email, rating, testimonial, status) VALUES (?, ?, ?, ?, ?, 'pending')";
    $stmt = $pdo->prepare($sql);
    
    // Execute the statement
    $result = $stmt->execute([$name, $role, $email, $rating, $testimonial]);
    
    if ($result) {
        // Send success response
        echo json_encode([
            'success' => true,
            'message' => 'Feedback submitted successfully!'
        ]);
    } else {
        throw new Exception('Failed to insert data into database');
    }
    
} catch (PDOException $e) {
    // Database error
    error_log("Database Error: " . $e->getMessage());
    echo json_encode([
        'success' => false,
        'message' => 'Database error: ' . $e->getMessage()
    ]);
} catch (Exception $e) {
    // Validation or other error
    error_log("Error: " . $e->getMessage());
    echo json_encode([
        'success' => false,
        'message' => $e->getMessage()
    ]);
}
?> 