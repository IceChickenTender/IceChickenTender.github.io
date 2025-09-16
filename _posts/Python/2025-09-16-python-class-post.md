---
title: "[Python] Python 클래스 기초"
categories:
  - Python
tags:
  - Python
  
toc: true
toc_sticky: true
toc_label: "Python 클래스 기초"
---

이번엔 Python 클래스의 기초에 대해서 알아보도록 하겠습니다. 이론적 개념인 OOP 등에 대해서는 다루지 않으면 Python 의 클래스에 대해서만 알아보도록 하겠습니다. 포스트 작성에는 "혼자 공부하는 파이썬" 책을 참조하였습니다.

# 1. 클래스 기본

## 1.1 클래스 선언하기

Python 에서의 클래스의 기본적인 구문을 살펴보고 무엇이 효율적인지 알아보도록 하겠습니다. Python 에서 클래스는 다음과 같은 구문으로 생성합니다.

```python
class 클래스_이름:
	클래스_내용
```

클래스를 이용해 학생 정보를 담는 `Student` 라는 클래스를 생성해 보겠습니다.

```python
# 클래스를 선언합니다.
class Student:
	pass

# 학생을 선언합니다.
student = Student()

# 학생 리스트를 선언합니다.

students = [
	Student(),
	Student(),
	Student(),
	Student(),
	Student(),
	Student(),
	Student()
]
```

## 1.2 생성자

클래스 이름과 같은 함수를 생성자(constructor)라고 부릅니다. 클래스 내부에 `__init__` 이라는 함수를 만들면 객체를 생성할 때 처리할 내용을 작성할 수 있습니다.

```python

class 클래스_이름:
	def __init__(self, 추가적인_매개변수):
		pass

```

클래스 내부의 함수는 첫 번째 매개변수로 반드시 `self` 를 입력해야 합니다. 이 때 `self` 는 자기 자신을 나타내는 딕셔너리라고 생각하면 됩니다. 다만 `self` 가 가지고 있는 속성과 기능에 접근할 때는 `self.<식별자>` 형태로 접근합니다. 그럼 예제로 좀 더 구체적으로 알아보도록 하겠습니다.

```python
# 클래스를 선언합니다

class Student:
	def __init__(self, name, korean, math, english, science):
		self.name = name
		self.korean = korean
		self.math = math
		self.english = english
		self.science = science

# 학생 리스트를 선언합니다.

students = [
	Student("윤인성", 87, 98, 88, 95),
	Student("연하진", 92, 98, 96, 98),
	Student("구지연", 76, 96, 94, 90),
	Student("나선주", 98, 92, 96, 92),
	Student("윤아린", 95, 88, 98, 98),
	Student("윤명월", 64, 88, 92, 92)
]

# Student 인스턴스의 속성에 접근하는 방법
print(students[0].name)
print(students[0].korean)
print(students[0].math)
print(students[0].english)
print(students[0].science)
```

```
실행 결과

윤인성
87
98
88
95
```

이렇게 하면 Student 인스턴스가 생성될 때 속성이 추가됩니다.

## 1.3 메서드

클래스가 가지고 있는 함수를 메서드(method)라고 부릅니다. 클래스 내부에 메서드를 만들 때는 다음과 같이 사용합니다. 생성자를 선언하는 방법과 같습니다. 다만 첫 번째 매개변수로 self 넣어야 한다는 것을 잊지 말아주세요.

```python
class 클래스_이름:
	def 메서드_이름(self, 추가적인_매개변수):
		pass
```

그럼 예제로 구체적으로 알아보도록 하겠습니다.

```python
# 클래스를 선언합니다.

class Student:
	def __init__(self, name, korean, math, english, science):
		self.name = name
		self.korean = korean
		self.math = math
		self.english = english
		self.science = science

	def get_sum(self):
		return self.korean + self.math + \
			self.english + self.science

	def get_average(self):
		return self.get_sum() / 4

	def to_string(self):
		return "{}\t{}\t{}".format(
				self.name,
				self.get_sum(),
				self.get_average()
			)

# 학생 리스트를 선언합니다.

students = [
	Student("윤인성", 87, 98, 88, 95),
	Student("연하진", 92, 98, 96, 98),
	Student("구지연", 76, 96, 94, 90),
	Student("나선주", 98, 92, 96, 92),
	Student("윤아린", 95, 88, 98, 98),
	Student("윤명월", 64, 88, 92, 92)
]

# 학생을 한 명씩 반복합니다.

print("이름", "총점", "평균", sep="\t")
for student in students:
	print(student.to_string())
```

```
실행 결과

이름	총점	평균
윤인성	368	92.0
연하진	384	96.0
구지연	356	89.0
나선주	378	94.5
윤아린	379	94.75
윤명월	336	84.0
```

# 2. 클래스의 추가적인 구문

클래스를 사용하는 것은 작정하고 속성과 기능을 가진 객체를 만들겠다는 의미입니다. 그래서 Python 은 그에 따른 부가적인 기능을 지원합니다. 예를 들어 어떤 클래스를 기반으로 그 속성과 기능을 물려받아 새로운 클래스를 만드는 상속, 이러한 상속 관계에 따라서 객체가 어떤 클래스를 기반으로 만들었는지 확인할 수 있게 해주는 `isinstance()` 함수, Python 이 기본적으로 제공하는 `str()` 함수 혹은 연산자를 사용해서 클래스의 특정 함수를 호출할 수 있게 해주는 기능 등이 대표적인 예입니다. 이번에는 이러한 부가적인 기능에 대해서 알아보겠습니다.

## 2.1 어떤 클래스의 인스턴스인지 확인하기

일단 객체가 어떤 클래스로부터 만들어졌는지 확인할 수 있도록 `isinstance()` 함수를 제공합니다. `isinstance()` 함수는 첫 번째 매개변수에 객체, 두 번째 매개변수에 클래스를 입력합니다.

```python
isinstance(인스턴스, 클래스)
```

이 때 인스턴스가 해당 클래스를 기반으로 만들어졌다면 True, 전혀 상관 없는 인스턴스와 클래스라면 False 를 리턴합니다. 간단한 예제를 살펴보겠습니다.

```python
# 클래스를 선언합니다.

class Student:
	def __init__(self):
		pass

# 학생을 선언합니다.
student = Student()

# 인스턴스 확인하기
print("isinstance(student, Student):", isinstance(student, Student))
```

```
실행 하기

isinstance(student, Student): True
```

`isinstance()` 함수는 다양하게 활용할 수 있는 기능입니다. 간단한 예로 하나의 리스트 내부에 여러 종류의 인스턴스가 들어 있을 때, 인스턴스들으 구분하며 속성과 기능을 사용할 때 사용합니다.

다음 코드를 살펴봅시다. `Student` 와 `Teacher` 라는 클래스를 생성하고 `classroom` 이라는 리스트 내부에 객체들을 넣었습니다. 그리고 반복을 적용했을 때 요소가 `Student` 클래스의 인스턴스인지, `Teacher` 클래스의 인스턴스인ㅇ지 확인하고 각각의 대상이 가지고 있는 적절한 함수를 호출합니다.

```python
# 학생 클래스를 선언합니다.
class Student:
	def study(self):
		print("공부를 합니다")

# 선생님 클래스를 선업합니다.
class Teacher:
	def teach(self):
		print("학생을 가르칩니다")

# 교실 내부의 객체 리스트를 생성합니다.
classroom = [Student(), Student(), Teacher(), Student(), Student()]

# 반복을 적용해서 적절한 함수를 호출하게 합니다.
for person in classroom:
	if isinstance(person, Student):
		person.study()
	elif isinstance(person, Teacher):
		person.teach()
```

```
실행 결과

공부를 합니다
공부를 합니다
학생을 가르칩니다
공부를 합니다
공부를 합니다
```

## 2.2 특수한 이름의 메소드

우리가 만든 `Student` 클래스를 기반으로 객체를 만들고 객체 뒤에 .(마침표)를 이볅해서 자동 완성 기능을 살펴보면 우리가 만들지 않았던 함수들이 잔뜩 들어 있는 것을 확인할 수 있습니다.

<div align="center">
  <img src="/assets/images/python/python-class/special_name_method.png" width="50%" height="40%"/>
</div>

이는 모두 Python 이 클래스를 사용할 때 제공해 주는 보조 기능입니다. 그런데 이름들이 조금 특이합니다. `__<이름>__()` 형태로 되어 있습니다. 이러한 메서드는 특수한 상황에 자동으로 호출되도록 만들어 졌습니다.

우선 다음과 같이 `__str__()` 을 클래스 내부에 정의해 보겠습니다. 이렇게 `__str__()` 함수를 정의하면 `str()` 함수를 호출할 때 `__str__()` 함수가 자동으로 호출됩니다.

```python
# 클래스를 선언합니다.

class Student:
	def __init__(self, name, korean, math, english, science):
		self.name = name
		self.korean = korean
		self.math = math
		self.english = english
		self.science = science

	def get_sum(self):
		return self.korean + self.math + \
			self.english + self.science

	def get_average(self):
		return self.get_sum() / 4

	def __str__(self):
		return "{}\t{}\t{}".format(
				self.name,
				self.get_sum(),
				self.get_average()
			)

# 학생 리스트를 선언합니다.
students = [
	Student("윤인성", 87, 98, 88, 95),
	Student("연하진", 92, 98, 96, 98),
	Student("구지연", 76, 96, 94, 90),
	Student("나선주", 98, 92, 96, 92),
	Student("윤아린", 95, 88, 98, 98),
	Student("윤명월", 64, 88, 92, 92)
]

# 출력합니다.
print("이름", "총점", "평균", sep="\t")
for student in students:
	print(str(student)) # str() 함수의 매개변수로 넣으면 Student 의 __str__() 함수가 호출됩니다.
```

```
실행 결과

이름	총점	평균
윤인성	368	92.0
연하진	384	96.0
구지연	356	89.0
나선주	378	94.5
윤아린	379	94.75
윤명월	336	84.0
```

## 2.3 클래스 변수와 메서드

인스턴스가 속성과 기능을 가질 수도 있지만, 클래스가 속성과 기능을 가질 수도 있습니다. 이에 대해 살펴보도록 하겠습니다.

### 클래스 변수

클래스 변수를 만드는 방법부터 살펴보도록 하겠습니다. 클래스 변수는 `class` 구문 바로 아래의 단계에 변수를 선언하기만 하면 됩니다. 이렇게 만들어진 클래스 변수는 다음과 같이 사용합니다.

```python
# 클래스 변수 만들기
class 클래스_이름:
	클래스_변수 = 값
```

```python
# 클래스 변수에 접근하기
클래스_이름.변수_이름
```

그냥 클래스가 가지고 있는 변수이므로, 사용 방법은 일반 변수와 다르지 않습니다. 간단하게 학생의 수를 세는 `Student.count` 라는 변수를 만들고 활용해 봅시다.

```python
# 클래스를 선언합니다.
class Student:
	count = 0

	def __init__(self, name, korean, math, english, science):
		# 인스턴스 변수 초기화
		self.name = name
		self.korean = korean
		self.math = math
		self.english = english
		self.science = science

		# 클래스 변수 설정
		Student.count += 1
		print("{}번째 학생이 생성되었습니다.".format(Student.count))

# 학생 리스트를 선언합니다.
students = [
	Student("윤인성", 87, 98, 88, 95),
	Student("연하진", 92, 98, 96, 98),
	Student("구지연", 76, 96, 94, 90),
	Student("나선주", 98, 92, 96, 92),
	Student("윤아린", 95, 88, 98, 98),
	Student("윤명월", 64, 88, 92, 92)
]

# 출력합니다.
print()
print("현재 생성된 총 학생 수는 {}명입니다.".format(Student.count))
```

```
실행 결과

1번째 학생이 생성되었습니다.
2번째 학생이 생성되었습니다.
3번째 학생이 생성되었습니다.
4번째 학생이 생성되었습니다.
5번째 학생이 생성되었습니다.
6번째 학생이 생성되었습니다.

현재 생성된 총 학생 수는 6명입니다.
```

사실 일반적인 변수로 만드나 클래스 변수로 만드나 사용에는 큰 차이가 없으나 "클래스가 가진 기능"을 명시적으로 나타내서 변수를 만든다는 것이 포인트라고 할 수 있습니다.

### 클래스 함수

클래스 함수도 클래스 변수처럼 그냥 클래스가 가진 함수입니다. 일반적인 함수로 만드나 클래스 함수로 만드나 사용에는 큰 차이가 없습니다. 다만 "클래스가 가진 기능"이라고 명시적으로 나타내는 것뿐입니다.

```python
# 클래스 함수 만들기
class 클래스_이름:
	@classmethod
	def 클래스_함수_이름(cls, 매개변수):
		pass
```

클래스 함수의 첫 번째 매개변수에는 클래스 자체가 들어옵니다. 일반적으로 `cls` 라는 이름의 변수로 선언하며, 이렇게 만들어진 클래스 함수는 다음과 같이 사용합니다.

```python
# 클래스 함수 호출하기
클래스_이름.함수_이름(매개변수)
```

그럼 간단하게 사용해 봅시다. `students` 라는 학생 리스트를 아예 클래스 내부에 만들어 버리고 학생 리스트를 전부 출력하는 `Studnet.print()` 함수를 만들었습니다.

```python
# 클래스를 선언합니다.
class Student:
	# 클래스 변수
	count = 0
	students = []

	# 클래스 함수
	@classmethod
	def print(cls):
		print("------ 학생 목록 ------")
		print("이름\t총점\t평균")
		for student in cls.students:
			print(str(student))
		print("------ ------ ------")


	def __init__(self, name, korean, math, english, science):
		# 인스턴스 변수 초기화
		self.name = name
		self.korean = korean
		self.math = math
		self.english = english
		self.science = science
		Student.count += 1
		Student.students.append(self)

	def get_sum(self):
		return self.korean + self.math + \
			self.english + self.science

	def get_average(self):
		return self.get_sum() / 4

	def __str__(self):
		return "{}\t{}\t{}".format(
				self.name,
				self.get_sum(),
				self.get_average()
			)

# 학생 리스트를 선언합니다.
students = [
	Student("윤인성", 87, 98, 88, 95),
	Student("연하진", 92, 98, 96, 98),
	Student("구지연", 76, 96, 94, 90),
	Student("나선주", 98, 92, 96, 92),
	Student("윤아린", 95, 88, 98, 98),
	Student("윤명월", 64, 88, 92, 92)
]

# 현재 생성된 학생을 모두 출력합니다.
Student.print()
```

```
실행 결과

------ 학생 목록 ------
이름	총점	평균
윤인성	368	92.0
연하진	384	96.0
구지연	356	89.0
나선주	378	94.5
윤아린	379	94.75
윤명월	336	84.0
------ ------ ------
```

# 마치며

Python 의 클래스에 대한 기초적인 내용에 대해 알아보았습니다. 내용 중 잘못된 내용이나 오타, 궁금하신 것이 있으시면 댓글 달아주시기 바랍니다. 긴 글 읽어주셔서 감사합니다.