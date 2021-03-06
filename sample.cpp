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

#define DIMENSIONS  1    //ヒストグラムの次元数
#define UNIFORM 1        //一様性に関するフラグ
#define ACCUMULATE 0    //計算フラグ
#define SCALE_SHIFT 0    //スケーリングした入力配列の要素に加える値
#define HISTOGRAM_WIDTH  720    //ヒストグラムを描く画像の横幅
#define HISTOGRAM_HEIGHT 100    //ヒストグラムを描く画像の縦幅
#define LINE_THICKNESS -1    //線の太さ
#define LINE_TYPE   8        //線の種類
#define SHIFT    0            //座標の小数点以下の桁を表すビット数
#define    ITERATIONS    3    //膨張、収縮の回数
#define    TRADIUS    50    //登録画像の物体までの距離


CvHistogram *histogram;
int histogramSize = 180;    //ヒストグラムに描写される縦棒の数

float range_0[] = { 0, 180 };    //ヒストグラムの範囲
float* ranges[] = { range_0 };    //ヒストグラム各次元の範囲を示す配列のポインタ
int hist1[180];            //ヒストグラムの値を入れる
int hist2[180];            //ヒストグラムの値を入れる
float hist_seiki1[180];    //正規化したヒストグラムを入れる
float hist_seiki2[180];    //正規化したヒストグラムを入れる
IplImage *histogramImage = NULL;    //ヒストグラム画像用IplImage
IplImage *heikatuImage = NULL;
IplImage *h_histogramImage = NULL;
IplImage *h_heikatuImage = NULL;


using namespace std;
double gheikin    = 0;            //ガウシアンフィルタをつくる時のμの値
double gbunsan = 1.0;        //ガウシアンフィルタをつくる時のσの値
int gosa;                    //判定の際の誤差の範囲
CvMoments moments1;            //重心を求める時のもの
CvMoments moments2;            //重心を求める時のもの

struct HANTEI
{
    int kekka;
    CvPoint point;
    double txbectol;
    double tybectol;
    int radius;
    int min;
    int max;
};

int main( int argc, char **argv ){
    int key = 0;                                //キー入力用の変数
    int kekka;
    vector<HANTEI> ht;
    
    CvScalar color1;                            //HSV表色系で表した色
    CvScalar color2;                            //HSV表色系で表した色
    CvScalar color3;                            //HSV表色系で表した色
    CvScalar color4;                            //HSV表色系で表した色
    
    CvSeq *contours = NULL;
    unsigned char h1;                        //H成分
    unsigned char s1;                        //S成分
    unsigned char v1;                        //V成分
    
    unsigned char h2;                        //H成分
    unsigned char s2;                        //S成分
    unsigned char v2;                        //V成分
    
    unsigned char h3;                        //H成分
    unsigned char s3;                        //S成分
    unsigned char v3;                        //V成分
    
    unsigned char h4;                        //H成分
    unsigned char s4;                        //S成分
    unsigned char v4;                        //V成分
    
    unsigned char p[3];                        //V成分
    
    char windowNameSource[] = "Source";        //元画像を表示するウィンドウの名前
    char windowNameButtai[] = "Buttai";        //元画像を表示するウィンドウの名前
    char windowNameBinary1[] = "Binary1";       //処理結果を表示するウィンドウの名前
    char windowNameBinary2[] = "Binary2";       //処理結果を表示するウィンドウの名前
    char windowNameBinary3[] = "Binary3";       //処理結果を表示するウィンドウの名前
    char windowNameHistogram1[] = "Histogram1";        //ヒストグラムを表示するウィンドウの名前
    char windowNameHistogram2[] = "Histogram2";
    char windowNameHistogram4[] = "Histogram3";
    char windowNameHistogram3[] = "kekka";        //比較するヒストグラムを表示するウィンドウの名前
    char windowNameObject[] = "Object";
    char windowNameCont[] = "Cont";
    char windowNameKouho[] = "Kouho";
    char windowNameMiru[] = "Miru";
    float max_value = 0;                    //ヒストグラムの最大値
    int bin_w;                            //ヒストグラムの縦棒の横幅
    int max_hue    = 0;                    //ヒストグラムが最大の時のHUEの値
    int max_hue2    = 0;                    //平滑後のヒストグラムが最大の時のHUEの値
    int max_i[180];                            //ヒストグラムを正規分布で表した時のHUEの値
    int h_max_i[180];                            //比較するヒストグラムを正規分布で表した時のHUEの値
    int max_i_s[180];                            //ヒストグラムを正規分布で表した時のHUEの値
    int h_max_i_s[180];                            //比較するヒストグラムを正規分布で表した時のHUEの値
    int max;                                    //物体の高さを求めるための物体の候補の最大値
    int min;                                //物体の高さを求めるための物体の候補の最小値
    double xbectol;                                    //角度を求める時に用いるxベクトル
    double ybectol;                                //角度を求める時に用いるyベクトル
    
    double total = 0;                        //ガウシアン作成の時の正規化するための値
    double gausian[5];                        //ガウシアンの割合
    double filter[5];                    //ガウシアンフィルタ
    vector<int>    mount;                //ヒストグラムの山の位置
    vector<int>    valley;                //ヒストグラムの谷の位置
    vector<int>    h_mount;                //比較するヒストグラムの山の位置
    vector<int>    h_valley;                //比較するヒストグラムの谷の位置
    vector<int> hanteip;                //判定(山)
    vector<int> hanteim;                //判定(谷)
    
    CvRect rect;                    //点列を包括する傾かない短形を求める
    
    //IplImage *frameImage;            //
    //キャプチャ画像用IplImage
    /*IplImage *source1Image = cvLoadImage( "image/kikakusho2.bmp", CV_LOAD_IMAGE_ANYDEPTH | CV_LOAD_IMAGE_ANYCOLOR );
     IplImage *frameImage = cvLoadImage( "image/kikakusho1.bmp", CV_LOAD_IMAGE_ANYDEPTH | CV_LOAD_IMAGE_ANYCOLOR );*/
    
                            //cvLoadImage ファイルから画像を読み込む
    IplImage *source1Image = cvLoadImage( "image/object24.bmp", CV_LOAD_IMAGE_ANYDEPTH | CV_LOAD_IMAGE_ANYCOLOR );
    IplImage *frameImage = cvLoadImage( "image/buttai(0.4bai).bmp", CV_LOAD_IMAGE_ANYDEPTH | CV_LOAD_IMAGE_ANYCOLOR );
                            //cvCreateImage ヘッダの作成とデータ領域の確保
    IplImage *source2Image = cvCreateImage( cvSize( (int)(source1Image->width), (int)(source1Image->height)), IPL_DEPTH_8U, 3 );
    
    
    if ( source1Image == NULL ) {
        //画像が見つからなかった場合
        printf( "画像が見つかりません\n" );
        return -1;
    }
    if ( frameImage == NULL ) {
        //画像が見つからなかった場合
        printf( "画像が見つかりません\n" );
        return -1;
    }
    
    //cvResize source2Imageに合わせsource1Imageを拡張・縮小
    cvResize( source1Image, source2Image, 1 );
    
    CvSize sizeOfImage1 = cvGetSize( source2Image );
    CvSize sizeOfImage2 = cvGetSize( frameImage );
    histogramImage = cvCreateImage( cvSize( HISTOGRAM_WIDTH, HISTOGRAM_HEIGHT ), IPL_DEPTH_8U, 3 );
    heikatuImage = cvCreateImage( cvSize( HISTOGRAM_WIDTH, HISTOGRAM_HEIGHT ), IPL_DEPTH_8U, 3 );
    
    //画像を生成する
    IplImage *hsvImage1 = cvCreateImage( sizeOfImage1, IPL_DEPTH_8U, 3 );            //HSV画像用IplImage
    IplImage *hsvImage3 = cvCreateImage( sizeOfImage1, IPL_DEPTH_8U, 3 );            //HSV画像用IplImage
    IplImage *hsvImage2 = cvCreateImage( sizeOfImage2, 8, 3 );            //HSV画像用IplImage
    IplImage *objectImage = cvCreateImage( sizeOfImage2, 8, 1 );            //オブジェクトの候補位置1IplImage
    IplImage *kouhoImage = cvCreateImage( sizeOfImage2, 8, 1 );            //オブジェクトの候補位置2IplImage
    IplImage *kouho2Image = cvCreateImage( sizeOfImage2, 8, 3 );            //オブジェクトの候補位置2IplImage
    IplImage *h_hsvImage = NULL;
    IplImage *h_hueImage = NULL;
    IplImage *h_saturationImage = NULL;
    IplImage *h_valueImage = NULL;
    IplImage *miruImage = cvCreateImage( sizeOfImage2, 8, 3 );
    IplImage *temp = cvCreateImage( sizeOfImage2, 8, 1 );            //一時保存用IplImage
    IplImage *label = cvCreateImage( sizeOfImage2,IPL_DEPTH_16S,1);            //ラベル結果保存用IplImage
    
    
    IplImage *hueImage = cvCreateImage( sizeOfImage1, IPL_DEPTH_8U, 1);            //色相(H)情報用IplImage
    IplImage *saturationImage = cvCreateImage( sizeOfImage1, IPL_DEPTH_8U, 1);    //彩度(S)情報用IplImage
    IplImage *valueImage = cvCreateImage( sizeOfImage1, IPL_DEPTH_8U, 1);        //明度(V)情報用IplImage
    IplImage *binary1Image = cvCreateImage( sizeOfImage1, IPL_DEPTH_8U, 1 );        //二値化情報用IplImage
    IplImage *binary2Image = cvCreateImage( sizeOfImage1, IPL_DEPTH_8U, 1 );        //二値化情報用IplImage
    IplImage *binary1 = cvCreateImage( sizeOfImage1, IPL_DEPTH_8U, 3 );        //二値化情報用IplImage
    IplImage *binary3Image = cvCreateImage( sizeOfImage1, IPL_DEPTH_8U, 3 );        //二値化情報用IplImage
    IplImage *contImage = cvCreateImage( sizeOfImage2, IPL_DEPTH_8U, 1);
    IplImage *cont2Image = cvCreateImage( sizeOfImage2, IPL_DEPTH_8U, 1);
    IplImage *cont3Image = cvCreateImage( sizeOfImage2, IPL_DEPTH_8U, 3);
    IplImage *kekkaImage = cvCreateImage( sizeOfImage2, IPL_DEPTH_8U, 3);
    
    CvMemStorage *storage = cvCreateMemStorage (0);
    
    //ウィンドウを生成する
    cvNamedWindow( windowNameSource, CV_WINDOW_AUTOSIZE );
    cvNamedWindow( windowNameButtai, CV_WINDOW_AUTOSIZE );
    cvNamedWindow( windowNameBinary1, CV_WINDOW_AUTOSIZE );
    cvNamedWindow( windowNameBinary2, CV_WINDOW_AUTOSIZE );
    cvNamedWindow( windowNameBinary3, CV_WINDOW_AUTOSIZE );
    cvNamedWindow( windowNameHistogram1, CV_WINDOW_AUTOSIZE );
    cvNamedWindow( windowNameHistogram2, CV_WINDOW_AUTOSIZE );
    cvNamedWindow( windowNameHistogram3, CV_WINDOW_AUTOSIZE );
    cvNamedWindow( windowNameHistogram4, CV_WINDOW_AUTOSIZE );
    cvNamedWindow( windowNameCont, CV_WINDOW_AUTOSIZE );
    cvNamedWindow( windowNameKouho, CV_WINDOW_AUTOSIZE );
    
    //cvShowImage( windowNameButtai, frameImage );
    //cvShowImage( windowNameSource, source2Image );
    
    //BGRからHSVに変換する
    cvCvtColor( source2Image, hsvImage1, CV_BGR2HSV );
    cvCvtColor( source2Image, hsvImage3, CV_BGR2HSV );
    
    for( int x = 0; x < binary1Image->width; x++ ){
        for( int y = 0; y < binary1Image->height; y++ ){
            color1 = cvGet2D( hsvImage1, y, x );
            h1 = color1.val[0];
            s1 = color1.val[1];
            v1 = color1.val[2];
            
            if(v1 >= 30){
                cvSetReal2D( binary1Image, y, x, 0 );
                hist1[ h1 ]++;
            }else{
                cvSetReal2D( binary1Image, y, x, 255 );
            }
        }
    }
    //cvShowImage( windowNameBinary1, binary1Image );
   
    
    for( int i = 0; i < 180; i++ ){
        if ( max_value < hist1[ i ]){
            max_value = hist1[ i ];
        }
    }
    for( int j = 0; j < 180; j++ ){
        hist_seiki1[j] = hist1[j]/max_value * histogramImage->height;
    }
    
    printf( "値の最大値は%f\n", max_value );
    
    //HSV画像をH,S,V画像に分ける
    cvSplit( hsvImage1, hueImage, saturationImage, valueImage, NULL );
    
    //ヒストグラム画像を白で初期化する
    cvSet( histogramImage, cvScalarAll( 360 ), NULL );
    cvSet( heikatuImage, cvScalarAll( 360 ), NULL );
    
    //ヒストグラムの縦棒の横幅を計算する
    bin_w = cvRound( ( double )histogramImage->width / histogramSize );
    
    //ヒストグラムの縦棒を描画する
    for ( int i = 0; i < histogramSize; i++ ){
        IplImage *henkanImage = cvCreateImage( cvSize(1, 1), IPL_DEPTH_8U, 3 );
        cvSet2D( henkanImage,0, 0, cvScalar(i,255, 255,0) );
        cvCvtColor( henkanImage, henkanImage ,CV_HSV2BGR );
        CvScalar henkan = cvGet2D( henkanImage, 0, 0);
        cvRectangle(
                    histogramImage,
                    cvPoint( i * bin_w, histogramImage->height ),
                    cvPoint( ( i+1 ) * bin_w, histogramImage->height -  hist_seiki1[i] ),
                    henkan,
                    LINE_THICKNESS,
                    LINE_TYPE,
                    SHIFT
                    );
        cvReleaseImage( &henkanImage );
    }
    
    //画像を表示する
    //cvShowImage( windowNameHistogram1, histogramImage );
   

    
    //ガウシアンフィルタを計算する
    for ( int i = -2; i < 3; i++)
    {
        gausian[ i+2 ] = (1/(sqrt( 2 * M_PI)*gbunsan))*exp(-pow((i - gheikin),2.0)/2*pow(gbunsan,2.0));
        //printf("%f\n",gausian[ i+2 ]);
        total += gausian[ i+2 ];
    }
    //printf("%f\n",total);
    
    
    
    for( int i = 0; i < 5; i++)
    {
        filter[i] = gausian[ i ]/ total;
        printf("%f\n",filter[ i ]);
    }
    
    for ( int i = 0; i < histogramSize; i++ )
    {
        if( hist_seiki1[max_hue] < hist_seiki1[i])
        {
            max_hue = i;
        }
    }
    printf("最大値の頻度を示す色相は%d\n" , max_hue );
    
    for( int x = 0; x < binary2Image->width; x++ ){
        for( int y = 0; y < binary2Image->height; y++ ){
            
            color3 = cvGet2D( hsvImage3, y, x );
            
            h3 = color3.val[0];
            s3 = color3.val[1];
            v3 = color3.val[2];
            
            if(max_hue- 5 <= h3 && h3 <= max_hue + 5
               && v3 > 50)
            {
                cvSetReal2D( binary2Image, y, x, 255 );
            }else{
                cvSetReal2D( binary2Image, y, x, 0 );
            }
        }
    }
    for( int y = 0; y < binary3Image->height; y++ ){
        for( int x = 0; x < binary3Image->width; x++ ){
            
            p[0] = binary3Image->imageData[binary3Image->widthStep * y + x * 3 ];
            p[1] = binary3Image->imageData[binary3Image->widthStep * y + x * 3 + 1 ];
            p[2] = binary3Image->imageData[binary3Image->widthStep * y + x * 3 + 2 ];
            
            binary3Image->imageData[binary3Image->widthStep * y + x * 3 ] = cvRound( 255 );
            binary3Image->imageData[binary3Image->widthStep * y + x * 3 + 1 ] = cvRound( 255 );
            binary3Image->imageData[binary3Image->widthStep * y + x * 3 + 2 ] = cvRound( 255 );
            
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
    cvCvtColor( binary1Image, binary1, CV_GRAY2BGR );
    //dhの表示？
    cvLine( binary1, cvPoint( gX1, gY1), cvPoint( hx,hy), cvScalar( 0,0,255), 2, 8, 0);
    //cvShowImage( windowNameBinary1, binary1 );
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
    //dmの表示？
    cvLine( binary3Image, cvPoint( gX1, gY1), cvPoint( hx1,hy1), cvScalar( 255,0,0), 2, 8, 0);
    //cvShowImage( windowNameBinary2, binary2Image );
    cvCopy(source2Image,binary3Image,binary2Image);
    //cvShowImage( windowNameBinary3, binary3Image );
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
        max_i_s[j] = ((double)max_i[j]/(double)max_i[max_hue2]) * histogramImage->height;
        IplImage *henkan2Image = cvCreateImage( cvSize(1, 1), IPL_DEPTH_8U, 3 );
        cvSet2D( henkan2Image,0, 0, cvScalar(j,255, 255,0) );
        cvCvtColor( henkan2Image, henkan2Image ,CV_HSV2BGR );
        CvScalar henkan2 = cvGet2D( henkan2Image, 0, 0);
        //矩形、あるいは塗りつぶされた矩形を描写
        cvRectangle(
                    heikatuImage,
                    cvPoint( j * bin_w, histogramImage->height ),
                    cvPoint( ( j+1 ) * bin_w, histogramImage->height - max_i[j] ),
                    henkan2,
                    LINE_THICKNESS,
                    LINE_TYPE,
                    SHIFT
                    );
        cvReleaseImage( &henkan2Image );
    }
    
    //cvShowImage( windowNameHistogram2, heikatuImage );
    
    //printf("%d\n" , max_hue2 );
    
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
    
    for( int l = 0; l < mount.size(); l++ )
    {
        printf("山の位置は%d,%d\n"    ,mount[ l ],max_i_s[mount[ l ]]);
    }
    
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
    
    for( int l = 0; l < valley.size(); l++ )
    {
        printf("谷の位置は%d,%d\n"    ,valley[ l ],max_i_s[valley[ l ]]);
    }
    
    gosa = ((mount.size() + valley.size())*2)+2;
    
    //printf("kokomadekita");
    //    cvWaitKey( 0 );
    
    //BGRからHSVに変換する
    cvCvtColor( frameImage, hsvImage2, CV_BGR2HSV );
    //cvZero(objectImage);
    
    for( int x = 0; x < objectImage->width; x++ ){
        for( int y = 0; y < objectImage->height; y++ ){
            
            color2 = cvGet2D( hsvImage2, y, x );
            h2 = color2.val[0];
            s2 = color2.val[1];
            v2 = color2.val[2];
            
            if( max_hue - 5 <= h2 && h2 <= max_hue + 5  && s2 > v2 * 0.5 && v2 > 50)
            {
                cvSetReal2D(objectImage, y, x, 255 );
            }
            else{
                cvSetReal2D(objectImage, y, x, 0 );
            }
        }
    }
    
    //膨張をITERATIONS回繰り返す
    cvDilate( objectImage, temp, NULL, ITERATIONS );
    //縮小をITERATIONS回繰り返す
    cvErode( temp, objectImage, NULL, ITERATIONS );
    
    //cvShowImage( windowNameObject, objectImage );

    
    //輪郭抽出する
    cvFindContours( objectImage, storage, &contours, sizeof(CvContour), CV_RETR_LIST, 2, cvPoint(0, 0));
    
    cvZero(kouhoImage);
    cvZero( cont2Image );
    
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
            CvHistogram *h_histogram;
            cvDrawContours( kouhoImage, seq, cvScalar(255),cvScalar(255), 0, -1,8,cvPoint(0,0));
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
            
            rect = cvBoundingRect( seq, 0 );
            cvCircle( contImage, cvPoint( gX2, gY2 ), radius3*hankei, cvScalar( 255 ), -1, 8 );
            cvCircle( cont2Image, cvPoint( gX2, gY2 ), radius3*hankei, cvScalar( 255 ), -1, 8 );
            cvLine( cont3Image, cvPoint( gX2, gY2), cvPoint( xh,yh), cvScalar( 255,0,0), 2, 8, 0);
            //cvShowImage( windowNameCont, cont3Image );
            txbectol = gX2 - xh;
            tybectol = gY2 - yh;
            h_hsvImage = cvCreateImage( cvSize(rect.width,rect.height), 8, 3 );
            h_hueImage = cvCreateImage( cvSize(rect.width,rect.height), 8, 1 );
            h_saturationImage = cvCreateImage( cvSize(rect.width,rect.height), 8, 1 );
            h_valueImage = cvCreateImage( cvSize(rect.width,rect.height), 8, 1 );
            h_histogramImage = cvCreateImage( cvSize( HISTOGRAM_WIDTH, HISTOGRAM_HEIGHT ), IPL_DEPTH_8U, 3 );
            h_heikatuImage = cvCreateImage( cvSize( HISTOGRAM_WIDTH, HISTOGRAM_HEIGHT ), IPL_DEPTH_8U, 3 );
            for( int x = 0; x < contImage->width; x++ ){
                for( int y = 0; y < contImage->height; y++ ){
                    if( cvGetReal2D( contImage, y, x ) == 255 )
                    {
                        color4 = cvGet2D( hsvImage2, y, x );
                        h4 = color4.val[0];
                        s4 = color4.val[1];
                        v4 = color4.val[2];
                        if( s4 > v4 * 0.5)
                        {
                            hist2[ h4 ]++;
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
                hist_seiki2[j] = hist2[j]/h_max_value * histogramImage->height;
            }
            //ヒストグラム画像を白で初期化する
            cvSet( h_histogramImage, cvScalarAll( 360 ), NULL );
            cvSet( h_heikatuImage, cvScalarAll( 360 ), NULL );
            //ヒストグラムの縦棒を描画する
            for ( int i = 0; i < histogramSize; i++ ){
                IplImage *h_henkanImage = cvCreateImage( cvSize(1, 1), IPL_DEPTH_8U, 3 );
                cvSet2D( h_henkanImage,0, 0, cvScalar(i,255, 255,0) );
                cvCvtColor( h_henkanImage, h_henkanImage ,CV_HSV2BGR );
                CvScalar h_henkan = cvGet2D( h_henkanImage, 0, 0);
                cvRectangle(
                            h_histogramImage,
                            cvPoint( i * bin_w, h_histogramImage->height ),
                            cvPoint( ( i+1 ) * bin_w, h_histogramImage->height - hist_seiki2[i] ),
                            h_henkan,
                            LINE_THICKNESS,
                            LINE_TYPE,
                            SHIFT
                            );
                cvReleaseImage( &h_henkanImage );
            }
            //画像を表示する
            //cvShowImage( windowNameHistogram4, h_histogramImage );
     
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
                h_max_i_s[j] = ((double)h_max_i[j]/(double)h_max_i[max_hue3]) *histogramImage->height;
                IplImage *henkan2Image = cvCreateImage( cvSize(1, 1), IPL_DEPTH_8U, 3 );
                cvSet2D( henkan2Image,0, 0, cvScalar(j,255, 255,0) );
                cvCvtColor( henkan2Image, henkan2Image ,CV_HSV2BGR );
                CvScalar henkan2 = cvGet2D( henkan2Image, 0, 0);
                cvRectangle(
                            h_heikatuImage,
                            cvPoint( j * bin_w, h_histogramImage->height ),
                            cvPoint( ( j+1 ) * bin_w, h_histogramImage->height - h_max_i_s[j]),
                            henkan2,
                            LINE_THICKNESS,
                            LINE_TYPE,
                            SHIFT
                            );
                cvReleaseImage( &henkan2Image );
            }
            //画像を表示する
            cvShowImage( windowNameHistogram4, h_heikatuImage );
            cvCopy( frameImage, miruImage, 0 );
            cvCircle( miruImage, cvPoint( gX2, gY2 ), radius3*hankei, cvScalar( 255 ), 1, 8 );
            cvShowImage( windowNameMiru, miruImage );
            
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
            
            for( int l = 0; l < h_mount.size(); l++ )
            {
                //printf("山の位置は%d,%d\n"    ,h_mount[ l ],h_max_i[h_mount[ l ]]);
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
            
            for( int l = 0; l < h_valley.size(); l++ )
            {
                //printf("谷の位置は%d,%d\n"    ,h_valley[ l ],h_max_i[h_valley[ l ]]);
            }
            //printf("終わりました\n");
            
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
            kekka = 0;
            
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
    
    cvCopy( frameImage, kouho2Image,kouhoImage);
    cvShowImage( windowNameKouho, kouho2Image );
    cvCopy( frameImage, kekkaImage );
    int bt =120;
    for( int j = 0; j < ht.size(); j++ )
    {
        if( bt > ht[j].kekka ){
            bt = ht[j].kekka;
        }
    }
    
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
            cvCircle( kekkaImage, ht[i].point, ht[i].radius, cvScalar( 255,255,255 ), 2, 8 );
            height = ht[i].max - ht[i].min;
            printf("画像中の高さは%d\n", height );
            a = (double)source1Image->height * TRADIUS;
            radius = a / (double)height;
            printf("物体までの距離は%f\n", radius );
            b =atan( ybectol / xbectol );
            c =atan( ht[i].tybectol / ht[i].txbectol);
            sign = (180/CV_PI)*(c - b);
            printf("物体までの角度は%f\n", sign );
            cvShowImage( windowNameHistogram3, kekkaImage );
        }
        
    }
    
    ht.clear();
    
    //キー入力を待つ
    cvWaitKey( 0 );
    
    //メモリを解放する
    cvReleaseImage( &source1Image );
    cvReleaseImage( &source2Image );
    cvReleaseImage( &binary1Image );
    cvReleaseImage( &binary2Image );
    cvReleaseImage( &binary3Image );
    cvReleaseImage( &hsvImage1 );
    cvReleaseImage( &hsvImage2 );
    cvReleaseImage( &h_hsvImage );
    cvReleaseImage( &hueImage );
    cvReleaseImage( &saturationImage );
    cvReleaseImage( &valueImage );
    cvReleaseImage( &histogramImage );
    cvReleaseImage( &h_histogramImage );
    cvReleaseImage( &objectImage );
    cvReleaseImage( &kouhoImage );
    cvReleaseImage( &heikatuImage );
    cvReleaseImage( &h_heikatuImage );
    cvReleaseImage( &h_histogramImage );
    cvDestroyWindow( windowNameSource );
    cvDestroyWindow( windowNameBinary1 );
    cvDestroyWindow( windowNameBinary2 );
    cvDestroyWindow( windowNameBinary3 );
    cvDestroyWindow( windowNameHistogram1 );
    cvDestroyWindow( windowNameHistogram2 );
    cvDestroyWindow( windowNameHistogram3 );
    cvDestroyWindow( windowNameHistogram4 );
    cvDestroyWindow( windowNameCont );
    cvDestroyWindow( windowNameObject );
    cvDestroyWindow( windowNameKouho );
    
    return 0;
}





























/* main
 int main(void)
 {
 IplImage *img;
 IplImage *gray;
 IplImage *color;
 int i,x,y;
 
 // 画像読み込み
 img = cvLoadImage("images/lena.jpg",CV_LOAD_IMAGE_UNCHANGED);
 if (img == NULL)
 {
 printf("Error loading file");
 return 0;
 }
 
 
 gray = cvCreateImage(cvGetSize(img),IPL_DEPTH_8U,1); // grayをimgの大きさに合わせ、グレー用として使用
 cvCvtColor(img,gray,CV_BGR2GRAY);// imgをgrayへ変換する
 cvThreshold(gray,gray,50,255,CV_THRESH_BINARY); // ２値化処理をする
 cvNot(gray, gray);
 
 color = cvCreateImage(cvGetSize(img), img->depth, img->nChannels);
 cvCopy(img, color, gray);
 //cvCvtColor(color, color, CV_BGR2HSV);// 表色系をBGRからHSVに変換する
 
 //ヒストグラムの抽出
 IplImage *src = color;
 IplImage* h_plane = cvCreateImage( cvGetSize(src), 8, 1 );
 IplImage* planes[] = { h_plane };
 IplImage* hsv = cvCreateImage( cvGetSize(src), 8, 3 );
 int h_bins = 180;
 int hist_size[] = {h_bins};
 float h_ranges[] = { 0, 180 };
 float* ranges[] = { h_ranges };
 CvHistogram* hist;
 float max_value = 0;
 int h;
 
 cvCvtColor(src, hsv, CV_BGR2HSV);
 cvCvtPixToPlane(hsv, h_plane, 0, 0, 0);
 hist = cvCreateHist(1, hist_size, CV_HIST_ARRAY, ranges, 1);
 cvCalcHist(planes, hist, 0, 0);
 cvGetMinMaxHistValue(hist, 0, &max_value, 0, 0);
 
 for( h = 0; h < h_bins; h++ )
 {
 float bin_val = cvQueryHistValue_1D(hist, h);
 //int intensity = cvRound(bin_val*255/max_value);
 printf("%f\n", bin_val);
 }
 
 
 cvNamedWindow("Window", 1); // ウィンドウの作成
 cvShowImage("Window",hsv);// 画像の表示
 cvWaitKey(0); // キー入力待機
 cvReleaseImage(&img);// メモリ解放
 cvReleaseImage(&gray);
 cvReleaseImage(&color);
 
 cvDestroyWindow("Window"); // ウィンドウを消す
 
 
 return 0;
 } */
