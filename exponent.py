# 거듭제곱 계산기

# 숫자와 지수를 입력받음
base = float(input("밑수를 입력하세요: "))
exponent = float(input("지수를 입력하세요: "))

# 거듭제곱 계산
result = base ** exponent

# 결과 출력
print(f"{base}의 {exponent} 거듭제곱은 {result}입니다.")