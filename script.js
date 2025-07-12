document.addEventListener('DOMContentLoaded', () => {
    const taskInput = document.getElementById('task-input');
    const priorityInput = document.getElementById('priority-input');
    const addTaskBtn = document.getElementById('add-task-btn');
    const taskList = document.getElementById('task-list');
    const taskListArea = document.querySelector('.task-list-area'); // For showing 'no tasks' message
    const tasksHeading = taskListArea.querySelector('h2');

    // Load tasks from local storage or use an empty array
    let tasks = JSON.parse(localStorage.getItem('tasks')) || [];

    // Function to save tasks to local storage
    const saveTasks = () => {
        localStorage.setItem('tasks', JSON.stringify(tasks));
    };

    // Function to render tasks to the DOM
    const renderTasks = () => {
        taskList.innerHTML = ''; // Clear existing tasks
        removeNoTasksMessage(); // Clear any existing "no tasks" message

        if (tasks.length === 0) {
            showNoTasksMessage();
            tasksHeading.style.display = 'none'; // Hide "Tasks" heading
            return;
        }
        tasksHeading.style.display = ''; // Show "Tasks" heading

        tasks.forEach(task => {
            const taskItem = document.createElement('li');
            taskItem.classList.add('task-item');
            if (task.completed) {
                taskItem.classList.add('completed');
            }
            taskItem.dataset.id = task.id;

            const checkbox = document.createElement('input');
            checkbox.type = 'checkbox';
            checkbox.classList.add('task-complete-checkbox');
            checkbox.checked = task.completed;
            checkbox.addEventListener('change', () => toggleComplete(task.id));

            const taskTextSpan = document.createElement('span');
            taskTextSpan.classList.add('task-text');
            taskTextSpan.textContent = task.text;

            const prioritySpan = document.createElement('span');
            prioritySpan.classList.add('task-priority-indicator');
            prioritySpan.textContent = task.priority.charAt(0).toUpperCase() + task.priority.slice(1);
            prioritySpan.classList.add(`priority-${task.priority}`);


            const editBtn = document.createElement('button');
            editBtn.classList.add('edit-btn');
            editBtn.textContent = 'Edit';
            editBtn.addEventListener('click', () => enableEditMode(taskItem, task));

            const deleteBtn = document.createElement('button');
            deleteBtn.classList.add('delete-btn');
            deleteBtn.textContent = 'Delete';
            deleteBtn.addEventListener('click', () => deleteTask(task.id));

            taskItem.appendChild(checkbox);
            taskItem.appendChild(taskTextSpan);
            taskItem.appendChild(prioritySpan);
            taskItem.appendChild(editBtn);
            taskItem.appendChild(deleteBtn);

            taskList.appendChild(taskItem);
        });
    };

    // Function to add a new task
    const addTask = () => {
        const text = taskInput.value.trim();
        const priority = priorityInput.value;

        if (text === '') {
            alert('Task description cannot be empty.');
            return;
        }

        const newTask = {
            id: Date.now().toString(), // Simple unique ID
            text: text,
            priority: priority,
            completed: false
        };

        tasks.push(newTask);
        saveTasks();
        renderTasks();

        taskInput.value = ''; // Clear input
        priorityInput.value = 'medium'; // Reset priority
    };

    // Function to toggle task completion
    const toggleComplete = (id) => {
        tasks = tasks.map(task =>
            task.id === id ? { ...task, completed: !task.completed } : task
        );
        saveTasks();
        renderTasks();
    };

    // Function to delete a task
    const deleteTask = (id) => {
        tasks = tasks.filter(task => task.id !== id);
        saveTasks();
        renderTasks();
    };

    // Function to enable editing mode for a task
    const enableEditMode = (taskItem, task) => {
        taskItem.classList.add('editing'); // Add class to disable hover etc.
        const taskTextSpan = taskItem.querySelector('.task-text');
        const currentText = task.text;

        const editInput = document.createElement('input');
        editInput.type = 'text';
        editInput.value = currentText;
        editInput.classList.add('edit-input');

        const currentPriority = task.priority;
        const priorityEditSelect = document.createElement('select');
        priorityEditSelect.classList.add('priority-edit-select'); // For potential styling
        ['low', 'medium', 'high'].forEach(p => {
            const option = document.createElement('option');
            option.value = p;
            option.textContent = p.charAt(0).toUpperCase() + p.slice(1);
            if (p === currentPriority) {
                option.selected = true;
            }
            priorityEditSelect.appendChild(option);
        });

        const saveBtn = document.createElement('button');
        saveBtn.textContent = 'Save';
        saveBtn.classList.add('edit-btn', 'save-btn'); // Re-use edit-btn style and add save-btn specific
        saveBtn.addEventListener('click', () => saveEditedTask(task.id, editInput.value, priorityEditSelect.value, taskItem));

        const cancelBtn = document.createElement('button');
        cancelBtn.textContent = 'Cancel';
        cancelBtn.classList.add('delete-btn'); // Re-use delete-btn style for cancel
        cancelBtn.style.backgroundColor = '#7f8c8d'; // A more neutral color for cancel
         cancelBtn.addEventListener('mouseover', function() { this.style.backgroundColor = '#6c7a7b'; });
         cancelBtn.addEventListener('mouseout', function() { this.style.backgroundColor = '#7f8c8d'; });
        cancelBtn.addEventListener('click', () => renderTasks()); // Just re-render to cancel

        // Replace elements
        taskItem.replaceChild(editInput, taskTextSpan);
        // Find the old priority span to replace it
        const oldPrioritySpan = taskItem.querySelector('.task-priority-indicator');
        if (oldPrioritySpan) {
            taskItem.replaceChild(priorityEditSelect, oldPrioritySpan);
        } else { // Should not happen if structure is correct
            taskItem.insertBefore(priorityEditSelect, taskItem.querySelector('.edit-btn'));
        }


        const oldEditBtn = taskItem.querySelector('.edit-btn');
        const oldDeleteBtn = taskItem.querySelector('.delete-btn');
        taskItem.replaceChild(saveBtn, oldEditBtn);
        taskItem.replaceChild(cancelBtn, oldDeleteBtn);

        editInput.focus();
        // Allow saving with Enter key
        editInput.addEventListener('keypress', (e) => {
            if (e.key === 'Enter') {
                saveEditedTask(task.id, editInput.value, priorityEditSelect.value, taskItem);
            }
        });
    };

    // Function to save an edited task
    const saveEditedTask = (id, newText, newPriority, taskItem) => {
        newText = newText.trim();
        if (newText === '') {
            alert('Task description cannot be empty.');
            // Restore original view or re-render
            renderTasks();
            return;
        }

        tasks = tasks.map(task =>
            task.id === id ? { ...task, text: newText, priority: newPriority } : task
        );
        saveTasks();
        renderTasks(); // Re-render the whole list to ensure UI consistency
        // No need to remove 'editing' class here as renderTasks rebuilds the item
    };

    // Helper function to show "No tasks" message
    const showNoTasksMessage = () => {
        const existingMessage = taskListArea.querySelector('.no-tasks-message');
        if (!existingMessage) {
            const messageElement = document.createElement('p');
            messageElement.classList.add('no-tasks-message');
            messageElement.textContent = 'No tasks yet. Add one above!';
            taskListArea.appendChild(messageElement);
        }
    };

    // Helper function to remove "No tasks" message
    const removeNoTasksMessage = () => {
        const messageElement = taskListArea.querySelector('.no-tasks-message');
        if (messageElement) {
            messageElement.remove();
        }
    };

    // Event listeners
    addTaskBtn.addEventListener('click', addTask);
    taskInput.addEventListener('keypress', (e) => {
        if (e.key === 'Enter') {
            addTask();
        }
    });

    // Initial render of tasks on page load
    renderTasks();
});
