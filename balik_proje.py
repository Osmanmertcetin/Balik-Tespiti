import cv2
import numpy as np
import os

print("Model okunuyor...")
net = cv2.dnn.readNet("yolov4-obj_last.weights","yolov4-obj.cfg")               #Eğittim yolo modelini cv2 ile okumasını sağladım.
print("Model başarıyla okundu.")
classes = []                                                                    #Sınıfları içinde tutması için oluşturduğum liste.
with open("obj.names", "r") as f:                                               #İçinde sınıf isimleri bulunan obj.names dosyasından sınıfları okuyarak classes adlı listeye ekledim.
    classes = [line.strip() for line in f.readlines()]
layer_names = net.getLayerNames()                                               #layer_names ve output_layers tanımlanması gerektiğinden getLayerNames ve getUnconnectedOutLayers                                                             
output_layers = [layer_names[i-1] for i in net.getUnconnectedOutLayers()]       #fonksiyonlarını kullanarak tanımladım.


with os.scandir("test") as imgPaths:                                            #test isimli klasörü tarayarak test için kullanılacak fotoğrafların yollarını aldım.
    for path in imgPaths:
        img_path = str(path)[11:-2]

        img = cv2.imread("./test/"+img_path)                                    #Aldığımızı fotoğraf yolunu parametre olarak vererek fotoğrafı okudum.
        if img is None:
            print("Fotoğraf düzgün bir şekilde okunamadı.")                     #Fotoğraf okumada bir hata olursa diye bilgi mesajı yazdım.


        if(img.shape[0]<500):
            img = cv2.resize(img, None, fx=1, fy=1.4)
        if(img.shape[0]>700 and img.shape[0]<1000):
            img = cv2.resize(img, None, fx=1, fy=0.6)
        if(img.shape[0]>=1000 and img.shape[0]<1500):                           #Bu kısımda okunan fotoğrafın çok büyük ya da çok küçük olmaması için fotoğrafın boyutuna göre
            img = cv2.resize(img, None, fx=1, fy=0.3)                           #fotoğrafı yeniden boyutlandırdım.
        if(img.shape[1]<700):
            img = cv2.resize(img, None, fx=1.4, fy=1)
        if(img.shape[1]>1000):
            img = cv2.resize(img, None, fx=0.6, fy=1)

        height, width, channels = img.shape                                     #Fotoğrafın yükseklik ve genişliğini aldım.

        #Nesnelerin tarandığı kısım.
        blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)     #Fotoğrafı blob formatına dönüştürdüm.
        net.setInput(blob)                                                      #input olarak blobu verdim.
        outs = net.forward(output_layers)                                       #Fotoğrafta bulunan nesnelerin başlangıç ve bitiş koordinatları, isimleri gibi bilgileri tutar.


        class_ids = []                                                          #Class id'leri tutan liste.
        confidences = []                                                        #Tanımlanan nesnelerin hangi oranla doğru olduğunu tutan liste.
        boxes = []                                                              #Kutuları tutan liste.

        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]                                   #Taranan nesnenin doğruluk oranını aldım
        
                if confidence > 0.5:                                            #Eğer doğruluk oranı 0.5'den büyükse bir nesne tespit edildiği anlamına geliyor.

                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)                       #Bulunan nesnenin x merkez, y merkez, yükseklik ve genişlik bilgilerini aldım.
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)


                    x = int(center_x - w / 2)                                   #Tespit edilen nesneyi gösterecek dikdörtgen için koordinatları oluşturdum.
                    y = int(center_y - h / 2)

                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))                       #Bulduğum değerleri listelere ekledim.
                    class_ids.append(class_id)

        indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)                #Gürültüyü azaltmak için kullandım(Aynı nesnenin birden fazla kez seçilmesini engeller).

        font = cv2.FONT_HERSHEY_PLAIN                                           #Yazı fontu.


        if(len(boxes)==0):                                                      #Eğer bir nesne tespit edilemediyse bilgi mesajı olarak fotoğrafa tespit edilemedi yazdırdım.
            cv2.putText(img, "Tespit edilemedi.", (30,30), font, 1, [255,255,255], 2)

        for i in range(len(boxes)):
            if i in indexes:
                x, y, w, h = boxes[i]                                           #Koordinat değerlerini aldım.
                label = str(classes[class_ids[i]])                              #Tespit edilen nesnenin ismini aldım.

                if(img_path[0:-6]==label[6:-1]):                                #Eğer fotoğraf ismi ile tespit edilen nesnenin ismi aynıysa yani yapılan tahmin doğruysa çizilecek dikdörtgen
                    color = [ 10.99937901, 186.2358544,   49.52151747]          #için renk değerini yeşil olarak ayarladım.
                else:
                    color = [0,0,255]                                           #Yapılan tahmin yanlışsa renk değerini kırmızı olarak ayarladım.


                cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)            #Nesneyi içine alan dikdörtgeni çizdirdim.
                cv2.putText(img, label, (x, y + 30), font, 1, [255,255,255], 2) #Tespit edilen nesnenin ismini yazdırdım.


        cv2.imshow(img_path, img)                                               #Fotoğrafı gösterdim.
        cv2.waitKey(0)
        cv2.destroyAllWindows()
