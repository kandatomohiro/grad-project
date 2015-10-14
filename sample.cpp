#include <opencv/cv.h>
#include <opencv/highgui.h>
#include <opencv/cxcore.h>

/*
#include で <> を使ってシステムにインストールした (brew install opencv でやったやつ) ライブラリを使う。
Xcode みたいにファイルをコピーして、どうこうしなくてよい。

実行する時は、

$ g++ sample.c `pkg-config --libs --cflags opencv`

^   ^        ^
|   |        |
|   |        +---- システムにインストールされている opencv のライブラリ
|   |f
|   +---- コンパイルしたいファイル (このファイル)
|
+---- コンパイルコマンド

すると、a.out っていうファイルができるから、

$ ./a.out

で実行すると、Window ができる。(他の window に隠れてるかもしてない)

このプログラムでは cvWaitKey(0) があるので、ESC キーで終了できるが、
ターミナル上で、Ctrl-C を押しても修了する。
*/


#include <stdio.h>
#include <math.h>

#include <iostream>
#include <vector>

using namespace std;

#define HISTOGRAM_WIDTH  720  // ヒストグラムを描く画像の横幅
#define HISTOGRAM_HEIGHT 100  // ヒストグラムを描く画像の縦幅
#define ITERATIONS       3    // 膨張、収縮の回数
#define TRADIUS          50   // 登録画像の物体までの距離

int histogramSize = 180;    //ヒストグラムに描写される縦棒の数

int hist1[180];          // ヒストグラムの値を入れる
int hist2[180];          // ヒストグラムの値を入れる
float hist_seiki1[180];  // 正規化したヒストグラムを入れる
float hist_seiki2[180];  // 正規化したヒストグラムを入れる

CvMoments moments1;  // 重心を求める時のもの
CvMoments moments2;  // 重心を求める時のもの

struct HANTEI {
    int kekka;
    CvPoint point;
    double txbectol;
    double tybectol;
    int radius;
    int min;
    int max;
};

int main( int argc, char **argv ){
    vector<HANTEI> ht;

    CvSeq *contours = NULL;

    int max_hue  = 0;  // ヒストグラムが最大の時のHUEの値
    int max_hue2 = 0;  // 平滑後のヒストグラムが最大の時のHUEの値

    int max_i[180];      // ヒストグラムを正規分布で表した時のHUEの値
    int h_max_i[180];    // 比較するヒストグラムを正規分布で表した時のHUEの値
    int max_i_s[180];    // ヒストグラムを正規分布で表した時のHUEの値
    int h_max_i_s[180];  // 比較するヒストグラムを正規分布で表した時のHUEの値
    int max;             // 物体の高さを求めるための物体の候補の最大値
    int min;             // 物体の高さを求めるための物体の候補の最小値
    double xbectol;      // 角度を求める時に用いるxベクトル
    double ybectol;      // 角度を求める時に用いるyベクトル

    double filter[5];  // ガウシアンフィルタ

    vector<int> mount;     // ヒストグラムの山の位置
    vector<int> valley;    // ヒストグラムの谷の位置
    vector<int> h_mount;   // 比較するヒストグラムの山の位置
    vector<int> h_valley;  // 比較するヒストグラムの谷の位置
    vector<int> hanteip;   // 判定(山)
    vector<int> hanteim;   // 判定(谷)

    // cvLoadImage ファイルから画像を読み込む
    IplImage *sourceImage = cvLoadImage("image/object24.bmp", CV_LOAD_IMAGE_ANYDEPTH | CV_LOAD_IMAGE_ANYCOLOR);
    IplImage *frameImage   = cvLoadImage("image/buttai(0.4bai).bmp", CV_LOAD_IMAGE_ANYDEPTH | CV_LOAD_IMAGE_ANYCOLOR);

    if (!sourceImage || !frameImage) {
        printf( "画像が見つかりません\n" );
        return -1;
    }

    CvSize sizeOfSourceImage = cvGetSize(sourceImage);
    CvSize sizeOfFrameImage  = cvGetSize(frameImage);

    // 画像を生成する
    IplImage *hsvImage1    = cvCreateImage(sizeOfSourceImage, IPL_DEPTH_8U, 3);  // HSV画像用IplImage
    IplImage *binary1Image = cvCreateImage(sizeOfSourceImage, IPL_DEPTH_8U, 1);  // 二値化情報用IplImage
    IplImage *binary2Image = cvCreateImage(sizeOfSourceImage, IPL_DEPTH_8U, 1);  // 二値化情報用IplImage

    IplImage *hsvImage2    = cvCreateImage(sizeOfFrameImage, IPL_DEPTH_8U, 3);   // HSV画像用IplImage
    IplImage *objectImage  = cvCreateImage(sizeOfFrameImage, IPL_DEPTH_8U, 1);   // オブジェクトの候補位置1IplImage
    IplImage *temp         = cvCreateImage(sizeOfFrameImage, IPL_DEPTH_8U, 1);   // 一時保存用IplImage
    IplImage *contImage    = cvCreateImage(sizeOfFrameImage, IPL_DEPTH_8U, 1);

    // BGRからHSVに変換する
    cvCvtColor(sourceImage, hsvImage1, CV_BGR2HSV);

    for (int x = 0; x < binary1Image->width; x++) {
        for (int y = 0; y < binary1Image->height; y++) {
            CvScalar color = cvGet2D(hsvImage1, y, x);
            unsigned char h = color.val[0];
            unsigned char v = color.val[2];

            if (v >= 30) {
                cvSetReal2D(binary1Image, y, x, 0);
                hist1[h]++;
            } else {
                cvSetReal2D(binary1Image, y, x, 255);
            }
        }
    }

    // ヒストグラムの最大値
    float max_value = 0;
    for (int i = 0; i < 180; i++) {
        if (max_value < hist1[i]) {
            max_value = hist1[i];
        }
    }
    for (int j = 0; j < 180; j++) {
        hist_seiki1[j] = hist1[j] / max_value * HISTOGRAM_HEIGHT;
    }

    printf("値の最大値は%f\n", max_value);

    //ガウシアンフィルタを計算する
    double gheikin = 0;    // ガウシアンフィルタをつくる時のμの値
    double gbunsan = 1.0;  // ガウシアンフィルタをつくる時のσの値
    double total   = 0;    // ガウシアン作成の時の正規化するための値
    double gausian[5];     // ガウシアンの割合

    for (int i = -2; i < 3; i++) {
        gausian[i + 2] = (1.0 / (sqrt(2.0 * M_PI) * gbunsan)) * exp(-pow((i - gheikin), 2.0) / 2.0 * pow(gbunsan, 2.0));
        total += gausian[i + 2];
    }
    for (int i = 0; i < 5; i++) {
        filter[i] = gausian[ i ] / total;
    }

    for ( int i = 0; i < histogramSize; i++ )
    {
        if( hist_seiki1[max_hue] < hist_seiki1[i])
        {
            max_hue = i;
        }
    }
    printf("最大値の頻度を示す色相は%d\n" , max_hue );

    for (int x = 0; x < binary2Image->width; x++) {
        for (int y = 0; y < binary2Image->height; y++) {
            CvScalar color = cvGet2D(hsvImage1, y, x);
            unsigned char h = color.val[0];
            unsigned char v = color.val[2];

            if (max_hue - 5 <= h && h <= max_hue + 5 && v > 50) {
                cvSetReal2D(binary2Image, y, x, 255);
            } else {
                cvSetReal2D(binary2Image, y, x, 0);
            }
        }
    }

    //重心を求める
    cvMoments( binary2Image, &moments1, 0 );
    double ma00 = cvGetSpatialMoment( &moments1, 0, 0 );
    double ma10 = cvGetSpatialMoment( &moments1, 1, 0 );
    double ma01 = cvGetSpatialMoment( &moments1, 0, 1 );
    int gX1 = ma10/ma00;
    int gY1 = ma01/ma00;
    int hx = 0;
    int hy = 0;
    int radius1 = 0;
    int radius2 = 0;

    for( int x = 0; x < binary2Image->width; x++ ){
        for( int y = 0; y < binary2Image->height; y++ ){
            if(cvGetReal2D( binary1Image, y, x ) ==0 )
            {
                double rad = (double) sqrt( pow(gX1 - x, 2.0 ) + pow( gY1 - y, 2.0 ) );
                if( radius1 <= (int) rad )
                {
                    radius1 = (int) rad;
                    hx = x;
                    hy = y;
                }
            }
        }
    }

    int hx1 = 0;
    int hy1 = 0;
    for( int x = 0; x < binary2Image->width; x++ ){
        for( int y = 0; y < binary2Image->height; y++ ){
            if(cvGetReal2D( binary2Image, y, x ) == 255 )
            {
                double rad = (double) sqrt( pow(gX1 - x, 2.0 ) + pow( gY1 - y, 2.0 ) );
                if( radius2 <= (int) rad )
                {
                    radius2 = (int) rad;
                    hx1 = x;
                    hy1 = y;
                }
            }
        }
    }

    xbectol = gX1 - hx1;
    ybectol = gY1 - hy1;
    double hankei = radius1 / radius2;
    printf("距離は%d,%d\n"    ,radius1,radius2);


    for( int j = 0; j < histogramSize; j++ ){
        if( j == 0 )
        {
            max_i[j] = cvRound( hist_seiki1[j+178]*filter[0] + hist_seiki1[j+179]*filter[1] + hist_seiki1[j]*filter[2] + hist_seiki1[j+1]*filter[3] + hist_seiki1[j+2]*filter[4] );
        }
        else if( j == 1 )
        {
            max_i[j] = cvRound( hist_seiki1[j+178]*filter[0] + hist_seiki1[j-1]*filter[1] + hist_seiki1[j]*filter[2] + hist_seiki1[j+1]*filter[3] + hist_seiki1[j+2]*filter[4] );
        }
        else if( j == 178 )
        {
            max_i[j] = cvRound( hist_seiki1[j-2]*filter[0] + hist_seiki1[j-1]*filter[1] + hist_seiki1[j]*filter[2] + hist_seiki1[j+1]*filter[3] + hist_seiki1[j-178]*filter[4] );
        }
        else if( j == 179 )
        {
            max_i[j] = cvRound( hist_seiki1[j-2]*filter[0] + hist_seiki1[j-1]*filter[1] + hist_seiki1[j]*filter[2] + hist_seiki1[j-179]*filter[3] + hist_seiki1[j-178]*filter[4] );
        }
        else
        {
            max_i[j] = cvRound( hist_seiki1[j-2]*filter[0] + hist_seiki1[j-1]*filter[1] + hist_seiki1[j]*filter[2] + hist_seiki1[j+1]*filter[3] + hist_seiki1[j+2]*filter[4] );
        }
    }

    for ( int i = 0; i < histogramSize; i++ )
    {
        if( max_i[max_hue2] < max_i[i])
        {
            max_hue2 = i;
        }
    }

    for( int j = 0; j < histogramSize; j++ ){
        max_i_s[j] = ((double)max_i[j]/(double)max_i[max_hue2]) * HISTOGRAM_HEIGHT;
    }

    for( int k = 0; k < histogramSize; k++ )
    {
        if( k == 0)
        {
            if ( max_i_s[k + 179] < max_i_s[k] && max_i_s[k] >= max_i_s[k + 1] && max_i_s[k]>20 )
            {
                mount.push_back( k );
            }
        }
        else
        {
            if( max_i_s[k - 1] < max_i_s[k] && max_i_s[k] >= max_i_s[k + 1]  && max_i_s[k]>20 )
            {
                mount.push_back( k );
            }
        }
    }

    /*
       for( int l = 0; l < mount.size(); l++ )
       {
       printf("山の位置は%d,%d\n"    ,mount[ l ],max_i_s[mount[ l ]]);
       }
       */

    for( int k = 0; k < histogramSize; k++ )
    {
        if( k == 0)
        {
            if ( max_i_s[k + 179] > max_i_s[k] && max_i_s[k] <= max_i_s[k + 1] && max_i_s[k]>20 )
            {
                valley.push_back( k );
            }
        }
        else
        {
            if( max_i_s[k - 1] > max_i_s[k] && max_i_s[k] <= max_i_s[k + 1]  && max_i_s[k]>20 )
            {
                valley.push_back( k );
            }
        }
    }

    /*
       for( int l = 0; l < valley.size(); l++ )
       {
       printf("谷の位置は%d,%d\n"    ,valley[ l ],max_i_s[valley[ l ]]);
       }
       */

    //BGRからHSVに変換する
    cvCvtColor( frameImage, hsvImage2, CV_BGR2HSV );
    //cvZero(objectImage);

    for (int x = 0; x < objectImage->width; x++) {
        for (int y = 0; y < objectImage->height; y++) {
            CvScalar color = cvGet2D(hsvImage2, y, x);
            unsigned char h = color.val[0];
            unsigned char s = color.val[1];
            unsigned char v = color.val[2];

            if (max_hue - 5 <= h && h <= max_hue + 5 && s > v * 0.5 && v > 50) {
                cvSetReal2D(objectImage, y, x, 255);
            } else {
                cvSetReal2D(objectImage, y, x, 0);
            }
        }
    }

    //膨張をITERATIONS回繰り返す
    cvDilate( objectImage, temp, NULL, ITERATIONS );
    //縮小をITERATIONS回繰り返す
    cvErode( temp, objectImage, NULL, ITERATIONS );

    //輪郭抽出する
    CvMemStorage *storage = cvCreateMemStorage (0);
    cvFindContours( objectImage, storage, &contours, sizeof(CvContour), CV_RETR_LIST, 2, cvPoint(0, 0));

    for ( CvSeq *seq = contours; seq; seq=seq->h_next )
    {
        double area = fabs(cvContourArea( seq,  CV_WHOLE_SEQ ));
        if( area >= 400 )
        {
            int max_hue3    = 0;                    //平滑後のヒストグラムが最大の時のHUEの値
            float h_max_value = 0;                    //比較するヒストグラムの最大値
            int radius3 = 0;
            double txbectol = 0;
            double tybectol = 0;
            for( int i = 0; i < 180; i++ )
            {
                hist2[i] = 0;
            }
            cvZero( contImage );
            cvDrawContours( contImage, seq, cvScalar(255), cvScalar(255), -1, CV_FILLED, 8);
            cvMoments( contImage, &moments2, 0 );
            double mb00 = cvGetSpatialMoment( &moments2, 0, 0 );
            double mb10 = cvGetSpatialMoment( &moments2, 1, 0 );
            double mb01 = cvGetSpatialMoment( &moments2, 0, 1 );
            int gX2 = mb10/mb00;
            int gY2 = mb01/mb00;
            int xh = 0;
            int yh = 0;

            min = contImage->height;
            max = 0;
            for( int x = 0; x < contImage->width; x++ ){
                for( int y = 0; y < contImage->height; y++ ){
                    if(cvGetReal2D( contImage, y, x ) == 255 )
                    {
                        double rad = (double) sqrt( pow(gX2 - x, 2.0 ) + pow( gY2 - y, 2.0 ) );
                        if( radius3 <= (int) rad )
                        {
                            radius3 = (int) rad;
                            xh = x;
                            yh = y;
                        }
                    }
                }
            }
            for( int x = 0; x < contImage->width; x++ ){
                for( int y = 0; y < contImage->height; y++ ){
                    if(cvGetReal2D( contImage, y, x ) == 255 )
                    {
                        if( y <= min)
                        {
                            min = y;
                        }
                        else if( y >= max )
                        {
                            max = y;
                        }
                    }
                }
            }

            txbectol = gX2 - xh;
            tybectol = gY2 - yh;

            cvCircle( contImage, cvPoint( gX2, gY2 ), radius3*hankei, cvScalar( 255 ), -1, 8 );
            for (int x = 0; x < contImage->width; x++) {
                for (int y = 0; y < contImage->height; y++) {
                    if (cvGetReal2D( contImage, y, x ) == 255) {
                        CvScalar color = cvGet2D(hsvImage2, y, x);
                        unsigned char h = color.val[0];
                        unsigned char s = color.val[1];
                        unsigned char v = color.val[2];

                        if (s > v * 0.5) {
                            hist2[h]++;
                        }
                    }
                }
            }
            for( int i = 0; i < 180; i++ ){
                if ( h_max_value < hist2[ i ]){
                    h_max_value = hist2[i];
                }
            }
            for( int j = 0; j < 180; j++ ){
                hist_seiki2[j] = hist2[j]/h_max_value * HISTOGRAM_HEIGHT;
            }

            for( int j = 0; j < histogramSize; j++ ){
                if( j == 0 )
                {
                    h_max_i[j] = cvRound( hist_seiki2[j+178]*filter[0] + hist_seiki2[j+179]*filter[1] + hist_seiki2[j]*filter[2] + hist_seiki2[j+1]*filter[3] + hist_seiki2[j+2]*filter[4] );
                }
                else if( j == 1 )
                {
                    h_max_i[j] = cvRound( hist_seiki2[j+178]*filter[0] + hist_seiki2[j-1]*filter[1] + hist_seiki2[j]*filter[2] + hist_seiki2[j+1]*filter[3] + hist_seiki2[j+2]*filter[4] );
                }
                else if( j == 178 )
                {
                    h_max_i[j] = cvRound( hist_seiki2[j-2]*filter[0] + hist_seiki2[j-1]*filter[1] + hist_seiki2[j]*filter[2] + hist_seiki2[j+1]*filter[3] + hist_seiki2[j- 178]*filter[4] );
                }
                else if( j == 179 )
                {
                    h_max_i[j] = cvRound( hist_seiki2[j-2]*filter[0] + hist_seiki2[j-1]*filter[1] + hist_seiki2[j]*filter[2] + hist_seiki2[j-179]*filter[3] + hist_seiki2[j- 178]*filter[4] );
                }
                else
                {
                    h_max_i[j] = cvRound( hist_seiki2[j-2]*filter[0] + hist_seiki2[j-1]*filter[1] + hist_seiki2[j]*filter[2] + hist_seiki2[j+1]*filter[3] + hist_seiki2[j+2]*filter[4] );
                }
            }
            for( int i = 0; i < histogramSize; i++ )
            {
                if( h_max_i[max_hue3] < h_max_i[i])
                {
                    max_hue3 = i;
                }
            }

            for( int j = 0; j < histogramSize; j++ )
            {
                h_max_i_s[j] = ((double)h_max_i[j]/(double)h_max_i[max_hue3]) * HISTOGRAM_HEIGHT;
            }


            for( int k = 0; k < histogramSize; k++ )
            {
                if( k == 0)
                {
                    if ( h_max_i_s[k + 179] < h_max_i_s[k] && h_max_i_s[k] >= h_max_i_s[k + 1] && h_max_i_s[k]>=14 )
                    {
                        h_mount.push_back( k );
                    }
                }
                else
                {
                    if( h_max_i_s[k - 1] < h_max_i_s[k] && h_max_i_s[k] >= h_max_i_s[k + 1]  && h_max_i_s[k]>=14 )
                    {
                        h_mount.push_back( k );
                    }
                }
            }


            for( int k = 0; k < histogramSize; k++ )
            {
                if( k == 0)
                {
                    if ( h_max_i_s[k + 179] > h_max_i_s[k] && h_max_i_s[k] <= h_max_i_s[k + 1] && h_max_i_s[k]>=20 )
                    {
                        h_valley.push_back( k );
                    }
                }
                else
                {
                    if( h_max_i_s[k - 1] > h_max_i_s[k] && h_max_i_s[k] <= h_max_i_s[k + 1]  && h_max_i_s[k]>=20 )
                    {
                        h_valley.push_back( k );
                    }
                }
            }

            for(int j = 0; j < mount.size(); j++)
            {
                hanteip.push_back(180);
                for(int i = 0; i < h_mount.size() ; i++ )
                {
                    if( mount[j] - h_mount[i] >= 0 )
                    {
                        if( hanteip[j] > mount[j] - h_mount[i])
                        {
                            hanteip[j] = mount[j] - h_mount[i];
                        }
                    }
                    else
                    {
                        if( hanteip[j] > -(mount[j] - h_mount[i]))
                        {
                            hanteip[j] = -(mount[j] - h_mount[i]);
                        }
                    }
                }
            }

            for( int k = 0; k < valley.size(); k++)
            {
                hanteim.push_back(0);
                for( int l = 0; l < h_valley.size(); l++ )
                {
                    if( valley[k] - h_valley[l] >= 0 )
                    {
                        if( hanteim[k] > valley[k] - h_valley[l])
                        {
                            hanteim[k] = valley[k] - h_valley[l];
                        }
                    }
                    else
                    {
                        if( hanteim[k] > -(valley[k] - h_valley[l]))
                        {
                            hanteim[k] = -(valley[k] - h_valley[l]);
                        }
                    }
                }
            }

            int kekka = 0;

            for( int i = 0; i < hanteip.size(); i++ )
            {
                kekka += hanteip[i];
            }

            for( int i = 0; i < hanteim.size(); i++ )
            {
                kekka += hanteim[i];
            }

            HANTEI t;

            t.kekka = kekka;
            t.point= cvPoint( gX2, gY2 );
            t.radius = radius3*hankei;
            t.min = min;
            t.max = max;
            t.txbectol = txbectol;
            t.tybectol = tybectol;
            ht.push_back( t );
            hanteip.clear();
            hanteim.clear();
            h_mount.clear();
            h_valley.clear();
        }
    }

    int bt =120;
    for( int j = 0; j < ht.size(); j++ )
    {
        if( bt > ht[j].kekka ){
            bt = ht[j].kekka;
        }
    }

    //ウィンドウを生成する
    char windowNameHistogram3[] = "kekka";  // 比較するヒストグラムを表示するウィンドウの名前
    cvNamedWindow( windowNameHistogram3, CV_WINDOW_AUTOSIZE );

    for( int i = 0; i < ht.size(); i++ )
    {
        if( ht[i].kekka == bt ){
            int height;
            double radius;
            double a;
            double b;
            double c;
            double sign;
            printf("描画します\n");
            cvCircle( frameImage, ht[i].point, ht[i].radius, cvScalar( 255,255,255 ), 2, 8 );
            height = ht[i].max - ht[i].min;
            printf("画像中の高さは%d\n", height );
            a = (double)sourceImage->height * TRADIUS;
            radius = a / (double)height;
            printf("物体までの距離は%f\n", radius );
            b =atan( ybectol / xbectol );
            c =atan( ht[i].tybectol / ht[i].txbectol);
            sign = (180/CV_PI)*(c - b);
            printf("物体までの角度は%f\n", sign );
            cvShowImage( windowNameHistogram3, frameImage );
        }

    }

    ht.clear();

    //キー入力を待つ
    cvWaitKey( 0 );

    //メモリを解放する
    cvReleaseImage( &sourceImage );
    cvReleaseImage( &binary1Image );
    cvReleaseImage( &binary2Image );
    cvReleaseImage( &hsvImage1 );
    cvReleaseImage( &hsvImage2 );
    cvReleaseImage( &objectImage );

    cvDestroyWindow( windowNameHistogram3 );
    return 0;
}
