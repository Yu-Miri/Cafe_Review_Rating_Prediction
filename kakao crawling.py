#필요한 library 설치
import time
import selenium
from selenium import webdriver
from bs4 import BeautifulSoup
from tqdm.notebook import tqdm
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys

def kakaomap_crawling(place):
    url = 'https://map.kakao.com/'
    options = webdriver.ChromeOptions() #--headless헤드리스
    options.add_argument('--headless') #크롬 창을 띄우지 않고 크롬을 사용하게 해주는 옵션
    options.add_argument('--no-sandbox') #크롬의 샌드박스 기능 비활성화
    options.add_argument('--disable-dev-shm-usage') #/deb/shm 디렉토리를 사용하지 않는다
    
    #설치한 웹 드라이버 파일 경로 옵션으로 넣고 불러오기
    driver = webdriver.Chrome(executable_path = '/Users/yumiri/Downloads/chromedriver_mac64/chromedriver.exe', options = options)

    #레이어 팝업창 때문에 url 두 번 불러오기
    for i in range(2):
        driver.get(url)


    # 찾는 부분이 하나면 element, 이상이면 elements
    search_bar = driver.find_element(By.ID, 'search.keyword.query')
    search_bar.send_keys(place+'카페' + '\n')

    #장소더보기 클릭
    place_clc = driver.find_element(By.XPATH, '//*[@id="info.search.place.more"]')
    place_clc.click()

    #리스트 생성
    rev_link = []

    #1-5번째 페이지 버튼 위치
    page_clc = driver.find_elements(By.XPATH, '/html/body/div[5]/div[2]/div[1]/div[7]/div[6]/div/a')

    #무한루프 돌리고 오류 발생 시 넘어가는 코드 작성
    try:
        while True:
            #페이지 버튼 한 번씩 돌아가면서 클릭
            for page in page_clc:
                page.click()
                #오류 방지를 위한 wait 초 시간 걸어주기
                driver.implicitly_wait(10)

                #뷰티풀숩 사용해서 페이지 소스를 html으로 바꿔서 soup 객체 지정
                soup = BeautifulSoup(driver.page_source,'html.parser')
                #가지고 올 href에 속한 elemehttps://map.kakao.com/#nt link 태그로 지정하고 포문으로 하나씩 추가
                link_tags = soup.select('a.numberofscore')
                for link_tag in link_tags:
                    #1-5페이지의 href를 리스트에 추가
                    rev_link.append(link_tag['href'])
    #             print(len(rev_link))
                #다음으로 넘어가는 클릭 버튼 눌러주기
            next_clc = driver.find_element(By.XPATH, '/html/body/div[5]/div[2]/div[1]/div[7]/div[6]/div/button[2]')
            next_clc.click()
            driver.implicitly_wait(10)
        rev_link = list(set(rev_link))
    except:
        pass

    #리스트 생성
    reviews, stars = [], []

    for rv_link in tqdm(rev_link):
        driver.get(rv_link) #for문 돌면서 링크 들어가기
        driver.implicitly_wait(20)

        #후기 더보기 클릭 버튼
        try:
            add_revs = driver.find_element(By.XPATH, '//*[@id="mArticle"]/div/div/a')
            driver.implicitly_wait(5)

            while driver.find_element(By.XPATH, '//*[@id="mArticle"]/div/div/a') == '후기 더보기':
                add_revs.click()
                time(5)

            sup = BeautifulSoup(driver.page_source, 'html.parser')
            review_tags = sup.select('p.txt_comment span')
            star_tags = sup.select('div.evaluation_review span.ico_star.inner_star')
            driver.implicitly_wait(5)

            for review_tag, star_tag in zip(review_tags, star_tags):
                reviews.append(review_tag.text)
                stars.append(star_tag['style'])


                #초기화
                driver.switch_to.default_content()
        except:
            pass
        
    return reviews, stars