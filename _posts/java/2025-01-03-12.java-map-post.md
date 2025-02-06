---
title: "[JAVA] JAVA Map 기초 정리"
categories:
  - JAVA
tags:
  - JAVA
toc: true
toc_sticky: true
toc_label: "JAVA Map 기초 정리"
---

이번엔 JAVA 의 Map 에 대해서 알아보도록 하겠습니다.   
Map 은 JAVA 에서 경우에 따라 List 보다 더 많이 사용될 수 있습니다. Map 은 중복이 허용되지 않는 다는 점, Key 와 Value 로 매핑하고자 하는 데이터를 저장하다는 점, 찾고자 하는 Value 값을 순차적으로 탐색하지 않고 Value 에 대응되는 Key 값을 통해 빠르게 요소를 탐색한다는 점 등 때문에 굉장히 많이 사용됩니다.   
따라서 Map 에 대해서 여러 기초적인 내용들을 이번 포스트에서 다뤄보도록 하겠습니다.

# Map 이란?

Map 은 key 와 Value 쌍의 집합을 나타내는 인터페이스로 사전(dictionary) 과 비슷합니다.   

# Map 의 특징

- Map 은 List 나 배열처럼 순차적으로(sequential) 요소값을 구하지 않고 Key 값을 이용해 Value 값을 얻습니다.
- Value 는 중복이 될 수 있지만 Key 는 중복이 될 수 없습니다.

# Map 을 사용하는 이유

- Key 와 Value 쌍의 사전 구조가 필요할 때

- 중복을 허용하지 않아야 할 때

- 요소의 삽입과 탐색 속도가 빨라야 할 때
  - 요소의 삽입과 검색의 시간 복잡도가 O(1) 이라 순차 구조인 List 나 배열에 비해 굉장히 빠름

# Map 의 종류와 사용 예시

1. HashMap : 가장 일반적으로 사용되는 구현체입니다.   
해시 함수를 사용해서 key-value 쌍을 저장하므로 매우 빠른 검색 속도를 제공하지만, 순서가 유지되지는 않습니다.

```java
Map<Class, Class> hashMap = new HashMap<>();
```

2. TreeMap : 이진 검색 트리를 기반으로 한 구현체입니다.   
key 의 순서를 유지하기 때문에 정렬된 상태로 데이터를 유지할 수 있습니다.

```java
Map<Class, Class> treeMap = new TreeMap<>();
```

3. LinkedHashMap : HashMap 과 LinkedList 를 결합한 구현체로 삽입 순서를 기억합니다.   
삽입 순서를 유지하면서 HashMap 의 기능을 사용할 수 있습니다.

```java
Map<Class, Class> linkedHashMap = new LinkedHashMap<>();
```

# Map 에서 자주 사용되는 메소드

아래 테이블에 `K` 는 key,`V` 는 value 로 보시면 됩니다.

|메소드명|리턴 타입|설명|
|:----:|:------:|:--:|
|clear()|void|map 에 있는 모든 요소들을 제거|
|containsKey(Object key)|boolean|map 의 key 값들 중에서 파라매터로 받은 key 값이 있는지 확인하는 메소드, map 에 해당 key 값이 있다면 true 를 반환|
|containsValue(Object value)|boolean|map 의 value 값들 중에서 파라매터로 받은 value 값이 있는지 확인하는 메소드, map 에 해당 value 값이 있다면 true 를 반환|
|equals(Object o)|boolean|map 의 값들 중에서 파라매터로 입력 받은 값과 일치하는 것이 있는지 확인하는 메소드, map 해당 값이 있다면 true 를 반환|
|get(Object key)|V|map 에서 파라매터로 입력 받은 key 값에 해당되는 value 값을 반환하는 메소드|
|isEmpty()|boolean|map 이 비었는지 확인하는 메소드|
|keySet()|Set<K>|map 의 key 값들이 요소인 Set 을 반환하는 메소드|
|put(K key, V value)|V|map 에 key 와 value 를 넣어주는 메소드|
|remove(Object key)|V|map 에서 파라매터로 받은 key 에 해당하는 entry 를 제거해 주는 메소드|
|size()|int|map 의 크기 값을 구하는 메소드|
|replace(K key, V value)|V|입력 파라매터의 key 가 map 에 있을 경우 해당 key 에 대응되는 value 값을 입력 파라매터로 받은 value 로 변경하고, 이전 value 값을 반환하는 메소드, 만약 이전 값이 없다면 null 을 반환|

# 마치며

이번엔 JAVA 의 map 기초에 대해서 알아보았습니다. map 은 프로그래밍을 함에 있어서 굉장히 중요한 데이터 구조 입니다. 이에 대해서 저는 두 가지 정도 이유가 생각나네요   

첫 번째 이유는 프로그래밍을 할 때에 가장 우선시 되어야 하는게 프로그램의 실행 시간인데 map 을 사용하게 되면 배열과 list 만 사용했을 때 보다 최소 몇 배는 빠르게 동작하도록 할 수 있습니다. 이는 map 이 배열과 list 와는 달리 요소를 찾을 때 요소들을 하나 하나 확인하는 것이 아니라 key 값만 주었을 때 대응되는 값을 바로 뱉어 내는 구조기 때문에 실행 시간이 O(1) 이기 때문입니다.   

두 번째 이유는 프로그래밍을 하다보면 우리는 데이터의 중복을 제거 해야 하는 케이스에 많이 직면하게 됩니다. 물론 코딩을 잘하시는 분들은 배열만으로도 충분히 실행 시간도 빠르면서 정확하게 중복을 제거하도록 할 수 있지만 매번 새로운 프로그램을 만들 때마다 이렇게 로직을 구현할 수도 없습니다 하지만 JAVA 에서 제공해 주는 map 은 데이터 중복을 허용하지 않기 때문에 이런 케이스에 대해 굉장히 유용하기 때문입니다.   

위 의 두 가지 이유 때문에 map 은 굉장히 중요하다고 볼 수 있고, 아마 이 포스트를 읽으시는 분들도 map 을 굉장히 많이 사용하게 되리라 생각이 됩니다.   
추후에는 이번 시간에 다루지 못했던 추가 기능이 존재하는 map 에 대해서 다뤄보도록 하겠으며 제가 모르는 map 의 정보나 기능들이 있다면 다뤄보도록 하겠습니다.   
긴 글 읽어주셔서 감사 드리며, 잘못된 내용, 오타, 궁금하신 내용이 있으시다면 댓글 달아주시기 바랍니다.