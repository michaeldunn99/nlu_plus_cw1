import heapq

top_scores = []  # Min-heap (stores tuples)

# Incoming scores with (score, name, favorite ice cream)
incoming_scores = [
    (55, "Alice", "Vanilla"),
    (72, "Bob", "Chocolate"),
    (88, "Charlie", "Strawberry"),
    (42, "David", "Mint"),
    (60, "Eve", "Cookie Dough"),
    (95, "Frank", "Rocky Road"),
    (85, "Grace", "Pistachio"),
    (77, "Hank", "Caramel"),
    (99, "Ivy", "Mango"),
    (58, "Jack", "Coffee"),
    (81, "Kelly", "Lemon"),
    (92, "Liam", "Cookies and Cream"),
    (70, "Mia", "Chocolate Chip"),
    (67, "Noah", "Banana"),
    (80, "Olivia", "Peach"),
]

# Keep only the top 5 highest scores
for score, player, flavor in incoming_scores:
    if len(top_scores) < 5:  # Keep only top 5 scores
        heapq.heappush(top_scores, (score, player, flavor))  # Push tuple
    elif score > top_scores[0][0]:  # Compare with the lowest score in heap
        heapq.heapreplace(top_scores, (score, player, flavor))  # Replace lowest

print("Heap contents (not sorted):", top_scores)  # Heap is NOT sorted

# Sort the heap in descending order for display
top_scores_sorted = sorted(top_scores, key=lambda x: x[0], reverse=True)
print("\nTop 5 Scores (Sorted Highest to Lowest):")
for score, player, flavor in top_scores_sorted:
    print(f"{player} - Score: {score}, Favorite Ice Cream: {flavor}")