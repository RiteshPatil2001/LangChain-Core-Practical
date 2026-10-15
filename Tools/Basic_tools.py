from langchain_core.tools import tool
from typing import TypedDict

class Student(TypedDict):
    id: int
    name: str
    standard: int
    subject: str

students : list[Student] = [
    {"id": 101, "name": "Rahul", "standard": 10, "subject": "Maths"},
    {"id": 102, "name": "Priya", "standard": 9, "subject": "Science"},
    {"id": 103, "name": "Amit", "standard": 10, "subject": "English"},
    {"id": 104, "name": "Sneha", "standard": 8, "subject": "History"},
    {"id": 105, "name": "Rohit", "standard": 9, "subject": "Geography"}
]

@tool
def getStudent(student_id: int) -> Student | None:
    """Return a student from student list on the basis of student id"""
    for student in students:
        if student["id"] == student_id:
            return student
    return None


result = getStudent.invoke({"student_id": 104})

print(result,"\n")
print(getStudent.name)
print(getStudent.description)
print(getStudent.args)
print(getStudent.args_schema.model_json_schema())


