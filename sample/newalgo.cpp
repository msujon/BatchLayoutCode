#include "newalgo.h"
#include "nblas.h"
#include "simd.h"

#define NEARZERO 1e-30
   

#define DEBUG 0 
   #define DOUT stderr     // debug output 

#define NO_FWRITE 1 
/*
 * MDIM selection 
 *
 */

   //#define MDIM_UR8_NOSYNC 1  // N=256: spup = 0.572 NDIM Efficient vecrion  
                              // N=128 spup = 0.634  FIXME: energy   
                              // N=64: spup = 0.683  
                              // N=32: spup = 0.974  

   //#define MDIM_UR4 1  // N=256: spup = 0.278   sec = 0.46  
                       // N=128: spup = 0.30    FIXED
                       // N=64:  spup = 0.33
                       // N=32:  spup = 0.44
                       // N=16:  spup = 0.68
                       // N=4:  spup = 1.71    sec = 0.43
   //#define MDIM_UR8 1  // FIXED   
   
   //#define MDIM_VEC_UR8 1  // N=256: spup = 1.46  FIXED  
                           // N=128: spup = 1.56
                           // N=64: spup = 1.85
   
   //#define MDIM_VEC_UR4 1  // N=256: spup = 1.35 FIXED 
                           // N=128: spup = 1.42  
                           // N=64: spup = 1.53 sec=0.099  
                           // N=32: spup = 2.03  --- with 256 Eff, spup = 1.26   
   
   //#define MDIM_VEC_UR8_NOSYNC 1  // N=256: spup = 1.93  ....  
                                  // N = 128: spup = 2.07   FIXED  
                                  // N = 64: spup = 2.27  
   
   #define MDIM_VEC_UR4_NOSYNC 1  // N=256: spup = 1.8 ... FIXED 
                                  // N=128: spup = 1.9
                                  // N=64: spup = 2.25 -- with 256 eff, spup = 1.85  
                                  // N=32: spup = 2.91 -- with 256 eff, spup = 1.89 
   //no splitting of two cases 
  //#define MDIM_VEC_UR4_MASK_NOSYNC 1  // N=256: 
                                       //

	newalgo::newalgo(CSR<INDEXTYPE, VALUETYPE> &A_csr, string input, string outputd, int init, double weight, double th, string ifile){
		graph.make_empty();
		graph = A_csr;
		outputdir = outputd;
		initfile = ifile;
		//cout << initfile << endl;
		W = weight;
		filename = input;
		threshold = th;
		nCoordinates = static_cast<Coordinate<VALUETYPE> *> (::operator new (sizeof(Coordinate<VALUETYPE>[A_csr.rows])));
		blasX = static_cast<VALUETYPE *> (::operator new (sizeof(VALUETYPE[A_csr.rows])));
                blasY = static_cast<VALUETYPE *> (::operator new (sizeof(VALUETYPE[A_csr.rows])));
		this->init = init;
	}
	void newalgo::randInit(){
		#pragma omp parallel for schedule(static)
		for(INDEXTYPE i = 0; i < graph.rows; i++){
			VALUETYPE x, y;
			do{
				x = -1.0 + 2.0 * rand()/(RAND_MAX+1.0);
				y = -1.0 + 2.0 * rand()/(RAND_MAX+1.0);
			}while(x * x + y * y > 1.0);
			x = x * MAXMIN;
			y = y * MAXMIN;
			nCoordinates[i] = Coordinate <VALUETYPE>(x, y, i);
		}
	}
	void newalgo::initDFS(){
                int visited[graph.rows] = {0};
                stack <int> STACK;
		double scalefactor = 1.0;
		minX = minY = 1.0;//numeric_limits<double>::max();
		maxX = maxY = 1.0;//numeric_limits<double>::min();
                STACK.push(0);
		double radi = 0.1;
                visited[0] = 1;
                nCoordinates[0] = Coordinate <VALUETYPE>(1.0, 1.0);
                while(!STACK.empty()){
                        int parent = STACK.top();
                        STACK.pop();
                        if(parent < graph.rows - 1 && (graph.rowptr[parent+1] - graph.rowptr[parent]) > 0){
                                double deg = 360.0 / (graph.rowptr[parent+1] - graph.rowptr[parent]);
                                double degree = 0;
                                for(INDEXTYPE n = graph.rowptr[parent]; n < graph.rowptr[parent+1]; n++){
                                        if(visited[graph.colids[n]] == 0){
                                                nCoordinates[graph.colids[n]] = Coordinate <VALUETYPE>(nCoordinates[parent].getX() + radi*cos(PI*(degree)/180.0), nCoordinates[parent].getY() + radi*sin(PI*(degree)/180.0)) + static_cast <float> (rand()) / static_cast <float> (RAND_MAX);
						visited[graph.colids[n]] = 1;
                                                STACK.push(graph.colids[n]);
                                                degree += deg;
						if(minX > nCoordinates[graph.colids[n]].getX()){
							minX = nCoordinates[graph.colids[n]].getX();
						}
						else if(maxX < nCoordinates[graph.colids[n]].getX()){
                                                        maxX = nCoordinates[graph.colids[n]].getX();
                                                }
						if(minY > nCoordinates[graph.colids[n]].getY()){
                                                        minY = nCoordinates[graph.colids[n]].getY();
                                                }
                                                else if(maxY < nCoordinates[graph.colids[n]].getY()){
                                                        maxY = nCoordinates[graph.colids[n]].getY();
                                                }
                                        }
                                }
                        }else{
				if((graph.nnz - graph.rowptr[parent]) <= 0) continue;

                                double deg = 360.0 / (graph.nnz - graph.rowptr[parent]);
                                double degree = 0;
                                for(INDEXTYPE n = graph.rowptr[parent]; n < graph.nnz; n++){
                                        if(visited[graph.colids[n]] == 0){
                                                nCoordinates[graph.colids[n]] = Coordinate <VALUETYPE>(nCoordinates[parent].getX() + radi*cos(PI*(degree)/180.0), nCoordinates[parent].getY() + radi*sin(PI*(degree)/180.0)) + static_cast <float> (rand()) / static_cast <float> (RAND_MAX);
						visited[graph.colids[n]] = 1;
                                                STACK.push(graph.colids[n]);
                                                degree += deg;
						if(minX > nCoordinates[graph.colids[n]].getX()){
                                                        minX = nCoordinates[graph.colids[n]].getX();
                                                }
                                                else if(maxX < nCoordinates[graph.colids[n]].getX()){
                                                        maxX = nCoordinates[graph.colids[n]].getX();
                                                }
                                                if(minY > nCoordinates[graph.colids[n]].getY()){
                                                        minY = nCoordinates[graph.colids[n]].getY();
                                                }
                                                else if(maxY < nCoordinates[graph.colids[n]].getY()){
                                                        maxY = nCoordinates[graph.colids[n]].getY();
                                                }
                                        }
                                }
                        }
                }
		scalefactor = 2.0 * MAXMIN / max(maxX - minX, maxY - minY);
		#pragma omp parallel for schedule(static)
		for(int i = 0; i < graph.rows; i++){
			nCoordinates[i] = nCoordinates[i] * scalefactor;
			blasX[i] = nCoordinates[i].x;
                        blasY[i] = nCoordinates[i].y;
		}
        }

	vector<VALUETYPE> newalgo::batchlayout(INDEXTYPE ITERATIONS, INDEXTYPE NUMOFTHREADS, INDEXTYPE BATCHSIZE){
		INDEXTYPE LOOP = 0;
                VALUETYPE start, end, ENERGY;
                VALUETYPE STEP = 1.0;
                vector<VALUETYPE> result;
		VALUETYPE (newalgo::*frm)(Coordinate<VALUETYPE>, Coordinate<VALUETYPE>);
		frm = &newalgo::frmodel;
                ENERGY = numeric_limits<VALUETYPE>::max();
                omp_set_num_threads(NUMOFTHREADS);
                start = omp_get_wtime();
                initDFS();
		NBLAS(BATCHSIZE, ITERATIONS, NUMOFTHREADS, nCoordinates, &ENERGY, graph, frm);
                end = omp_get_wtime();

                cout << "NBLAS Minibatch Size:" << BATCHSIZE  << endl;
                cout << "NBLAS Minbatch Energy:" << ENERGY << endl;
                cout << "NBLAS Minibatch Parallel Wall time required:" << end - start << endl;
                writeToFile("NBLAS"+ to_string(BATCHSIZE)+"PARAOUT" + to_string(LOOP));
                result.push_back(ENERGY);
                result.push_back(end - start);
		return result;
	}
	
	vector<VALUETYPE> newalgo::EfficientVersion(INDEXTYPE ITERATIONS, INDEXTYPE NUMOFTHREADS, INDEXTYPE BATCHSIZE){
                INDEXTYPE LOOP = 0;
                VALUETYPE start, end, ENERGY, ENERGY0, *pb_X, *pb_Y;
                VALUETYPE STEP = 1.0;
                vector<VALUETYPE> result;
                pb_X = static_cast<VALUETYPE *> (::operator new (sizeof(VALUETYPE[BATCHSIZE])));
                pb_Y = static_cast<VALUETYPE *> (::operator new (sizeof(VALUETYPE[BATCHSIZE])));
                ENERGY0 = ENERGY = numeric_limits<VALUETYPE>::max();
                omp_set_num_threads(NUMOFTHREADS);
                start = omp_get_wtime();
                initDFS();
                while(LOOP < ITERATIONS){
                        ENERGY0 = ENERGY;
                        ENERGY = 0;
			//#pragma imp parallel for schedule(static)
                        for(int i = 0; i < BATCHSIZE; i++){
                                pb_X[i] = pb_Y[i] = 0;
                        }
		
		//TODO: unrolling and jamming
		// no reverse

                        for(INDEXTYPE b = 0; b < (graph.rows / BATCHSIZE); b += 1){
                                #pragma omp parallel for schedule(static)
                                for(INDEXTYPE i = b * BATCHSIZE; i < (b + 1) * BATCHSIZE; i += 1){
                                        VALUETYPE fx = 0, fy = 0, distX, distY, dist, dist2;
                                        INDEXTYPE ind = i - b * BATCHSIZE;
					for(INDEXTYPE j = 0; j < i; j += 1){
                                                distX = blasX[j] - blasX[i];
                                                distY = blasY[j] - blasY[i];
                                                dist2 = 1.0 / (distX * distX + distY * distY);
                                                fx += distX * dist2;
                                                fy += distY * dist2;
						//if(j == graph.colids[graph.rowptr[i]])
						//printf("Reverse %d: x = %lf, y = %lf\n", j, distX*dist2, distY*dist2);
                                        }
					for(INDEXTYPE j = i+1; j < graph.rows; j += 1){
                                                distX = blasX[j] - blasX[i];
                                                distY = blasY[j] - blasY[i];
                                                dist2 = 1.0 / (distX * distX + distY * distY);
                                                fx += distX * dist2;
                                                fy += distY * dist2;
						//if(j == graph.colids[graph.rowptr[i]])
                                                //printf("Reverse %d: x = %lf, y = %lf\n", i, distX*dist2, distY*dist2);

                                        }
                                        for(INDEXTYPE j = graph.rowptr[i]; j < graph.rowptr[i+1]; j += 1){
                                                int v = graph.colids[j];
                                                distX = blasX[v] - blasX[i];
                                                distY = blasY[v] - blasY[i];
                                                dist = (distX * distX + distY * distY);
                                                dist = sqrt(dist) + 1.0 / dist;
                                                pb_X[ind] += distX * dist;
                                                pb_Y[ind] += distY * dist;
						//if(v == graph.colids[graph.rowptr[i]])
						//printf("Forward %d: x = %lf, y = %lf\n", v, distX * dist2, distY*dist2);
                                        }
                                        pb_X[ind] = pb_X[ind] - fx;
                                        pb_Y[ind] = pb_Y[ind] - fy;
                                }
                                for(INDEXTYPE i = b * BATCHSIZE; i < (b + 1) * BATCHSIZE; i++){
                                        INDEXTYPE ind = i - b * BATCHSIZE;
					double dist = (pb_X[ind]*pb_X[ind] + pb_Y[ind]*pb_Y[ind]);
                                        ENERGY += dist;
					dist = STEP / sqrt(dist);
					blasX[i] += pb_X[ind] * dist;
                                        blasY[i] += pb_Y[ind] * dist;
					pb_X[ind] = pb_Y[ind] = 0;
                                }
                        }
			//clean up loop
			INDEXTYPE cleanup = (graph.rows/BATCHSIZE) * BATCHSIZE;
			#pragma omp parallel for schedule(dynamic)
			for(INDEXTYPE i = cleanup; i < graph.rows; i += 1){
				INDEXTYPE ind = i- cleanup;
                		VALUETYPE fx = 0, fy = 0, distX, distY, dist, dist2;
				for(INDEXTYPE j = graph.rowptr[i]; j < graph.rowptr[i+1]; j += 1){
					int v = graph.colids[j];
                                        distX = blasX[v] - blasX[i];
                                        distY = blasY[v] - blasY[i];
                                        dist = distX * distX + distY * distY;
                                        dist = sqrt(dist) + 1.0 / dist;
                                        pb_X[ind] += distX * dist;
                                        pb_Y[ind] += distY * dist;	
				}
				for(INDEXTYPE j = 0; j < i; j += 1){
                                        distX = blasX[j] - blasX[i];
                                        distY = blasY[j] - blasY[i];
                                        dist2 = 1.0 / (distX * distX + distY * distY);
                                        fx += distX * dist2;
                                        fy += distY * dist2;
                                }
                               	for(INDEXTYPE j = i+1; j < graph.rows; j += 1){
                                        distX = blasX[j] - blasX[i];
                                        distY = blasY[j] - blasY[i];
                                        dist2 = 1.0 / (distX * distX + distY * distY);
                                        fx += distX * dist2;
                                        fy += distY * dist2;
                               	}
				pb_X[ind] = pb_X[ind] - fx;
				pb_Y[ind] = pb_Y[ind] - fy;
			}
			for(INDEXTYPE i = cleanup; i < graph.rows; i += 1){
				INDEXTYPE ind = i - cleanup;
                                double dist = (pb_X[ind]*pb_X[ind] + pb_Y[ind]*pb_Y[ind]);
                                ENERGY += dist;
				dist = STEP / sqrt(dist);
				blasX[i] += pb_X[ind] * dist;
                                blasY[i] += pb_Y[ind] * dist;
				pb_X[ind] = pb_Y[ind] = 0;
                       	}
                        STEP = STEP * 0.999;
                        LOOP++;
                }
		end = omp_get_wtime();
         #if NO_FWRITE == 0
                cout << "Efficient Minibatch Size:" << BATCHSIZE  << endl;
                cout << "Efficient Minbatch Energy:" << ENERGY << endl;
                cout << "Efficient Minibatch Parallel Wall time required:" << end - start << endl;
                writeToFileEFF("Efficient"+ to_string(BATCHSIZE)+"PARAOUT" + to_string(LOOP));
         #endif
                result.push_back(ENERGY);
                result.push_back(end - start);
                return result;
        }

/*
 *    Unrolled version 
 */
#if 0
	vector<VALUETYPE> newalgo::EfficientVersionUnRoll(INDEXTYPE ITERATIONS, INDEXTYPE NUMOFTHREADS, INDEXTYPE BATCHSIZE){
                INDEXTYPE LOOP = 0;
                VALUETYPE start, end, ENERGY, ENERGY0, *pb_X, *pb_Y;
                VALUETYPE STEP = 1.0;
                vector<VALUETYPE> result;
                pb_X = static_cast<VALUETYPE *> (::operator new (sizeof(VALUETYPE[BATCHSIZE])));
                pb_Y = static_cast<VALUETYPE *> (::operator new (sizeof(VALUETYPE[BATCHSIZE])));
                ENERGY0 = ENERGY = numeric_limits<VALUETYPE>::max();
                omp_set_num_threads(NUMOFTHREADS);
                start = omp_get_wtime();
                initDFS();
                while(LOOP < ITERATIONS){
                        ENERGY0 = ENERGY;
                        ENERGY = 0;
                        for(int i = 0; i < BATCHSIZE; i++){
                                pb_X[i] = pb_Y[i] = 0;
                        }
		
			for(INDEXTYPE b = 0; b < (graph.rows / BATCHSIZE); b += 1){
				#pragma omp parallel for schedule(static)	
				for(INDEXTYPE i = b * BATCHSIZE; i < (b + 1) * BATCHSIZE; i += 8){
                                        VALUETYPE fx0, fx1, fx2, fx3, fx4, fx5, fx6, fx7;
                                	VALUETYPE fy0, fy1, fy2, fy3, fy4, fy5, fy6, fy7;
                                	VALUETYPE x0, x1, x2, x3, x4, x5, x6, x7;
                                	VALUETYPE y0, y1, y2, y3, y4, y5, y6, y7;
                                	VALUETYPE d0, d1, d2, d3, d4, d5, d6, d7;
					
					int ind = i-b*BATCHSIZE;
					x0 = blasX[i];
					x1 = blasX[i+1];
					x2 = blasX[i+2];
					x3 = blasX[i+3];
					x4 = blasX[i+4];
					x5 = blasX[i+5];
					x6 = blasX[i+6];
					x7 = blasX[i+7];

					y0 = blasY[i];
					y1 = blasY[i+1];	
					y2 = blasY[i+2];
					y3 = blasY[i+3];
					y4 = blasY[i+4];
					y5 = blasY[i+5];
					y6 = blasY[i+6];
					y7 = blasY[i+7];
					
					fx0 = fx1 = fx2 = fx3 = fx4 = fx5 = fx6 = fx7 = 0;
					fy0 = fy1 = fy2 = fy3 = fy4 = fy5 = fy6 = fy7 = 0;		
                                       	VALUETYPE distX, distY, dist; 
					for(INDEXTYPE j = graph.rowptr[i]; j < graph.rowptr[i+1]; j += 1){
                                                int v = graph.colids[j];
                                                distX = blasX[v] - x0;
                                                distY = blasY[v] - y0;

                                                dist = (distX * distX + distY * distY);
                                                dist = sqrt(dist) + 1.0 / dist;


                                                fx0 += distX * dist;
                                                fy0 += distY * dist;
                                        }
					
					for(INDEXTYPE j = graph.rowptr[i+1]; j < graph.rowptr[i+1+1]; j += 1){
                                                int v = graph.colids[j];
                                                distX = blasX[v] - x1;
                                                distY = blasY[v] - y1;

                                                dist = (distX * distX + distY * distY);
                                                dist = sqrt(dist) + 1.0 / dist;


                                                fx1 += distX * dist;
                                                fy1 += distY * dist;
                                        }
				
					for(INDEXTYPE j = graph.rowptr[i+2]; j < graph.rowptr[i+1+2]; j += 1){
                                                int v = graph.colids[j];
                                                distX = blasX[v] - x2;
                                                distY = blasY[v] - y2;

                                                dist = (distX * distX + distY * distY);
                                                dist = sqrt(dist) + 1.0 / dist;


                                                fx2 += distX * dist;
                                                fy2 += distY * dist;
                                        }

					for(INDEXTYPE j = graph.rowptr[i+3]; j < graph.rowptr[i+1+3]; j += 1){
                                                int v = graph.colids[j];
                                                distX = blasX[v] - x3;
                                                distY = blasY[v] - y3;

                                                dist = (distX * distX + distY * distY);
                                                dist = sqrt(dist) + 1.0 / dist;


                                                fx3 += distX * dist;
                                                fy3 += distY * dist;
                                        }

					for(INDEXTYPE j = graph.rowptr[i+4]; j < graph.rowptr[i+1+4]; j += 1){
                                                int v = graph.colids[j];
                                                distX = blasX[v] - x4;
                                                distY = blasY[v] - y4;

                                                dist = (distX * distX + distY * distY);
                                                dist = sqrt(dist) + 1.0 / dist;


                                                fx4 += distX * dist;
                                                fy4 += distY * dist;
                                        }
					for(INDEXTYPE j = graph.rowptr[i+5]; j < graph.rowptr[i+1+5]; j += 1){
                                                int v = graph.colids[j];
                                                distX = blasX[v] - x5;
                                                distY = blasY[v] - y5;

                                                dist = (distX * distX + distY * distY);
                                                dist = sqrt(dist) + 1.0 / dist;


                                                fx5 += distX * dist;
                                                fy5 += distY * dist;
                                        }
					for(INDEXTYPE j = graph.rowptr[i+6]; j < graph.rowptr[i+1+6]; j += 1){
                                                int v = graph.colids[j];
                                                distX = blasX[v] - x6;
                                                distY = blasY[v] - y6;

                                                dist = (distX * distX + distY * distY);
                                                dist = sqrt(dist) + 1.0 / dist;


                                                fx6 += distX * dist;
                                                fy6 += distY * dist;
                                        }
					for(INDEXTYPE j = graph.rowptr[i+7]; j < graph.rowptr[i+1+7]; j += 1){
                                                int v = graph.colids[j];
                                                distX = blasX[v] - x7;
                                                distY = blasY[v] - y7;

                                                dist = (distX * distX + distY * distY);
                                                dist = sqrt(dist) + 1.0 / dist;


                                                fx7 += distX * dist;
                                                fy7 += distY * dist;
                                        }
					
					for(INDEXTYPE j = 0; j < i; j += 1){
                                        	VALUETYPE vd = blasX[j];
						x0 = vd - x0;
						x1 = vd - x1;
						x2 = vd - x2;
                                                x3 = vd - x3;
						x4 = vd - x4;
                                                x5 = vd - x5;
                                                x6 = vd - x6;
                                                x7 = vd - x7;
						y0 = vd - y0;
						y1 = vd - y1;
						y2 = vd - y2;
                                                y3 = vd - y3;
						y4 = vd - y4;
                                                y5 = vd - y5;
                                                y6 = vd - y6;
                                                y7 = vd - y7; 	       
						//distX = blasX[j] - blasX[i];
                                                //distY = blasY[j] - blasY[i];
                                         	
						d0 = 1.0 / (x0 * x0 + y0 * y0);
						d1 = 1.0 / (x1 * x1 + y1 * y1);
						d2 = 1.0 / (x2 * x2 + y2 * y2);
						d3 = 1.0 / (x3 * x3 + y3 * y3);
						d4 = 1.0 / (x4 * x4 + y4 * y4);
						d5 = 1.0 / (x5 * x5 + y5 * y5);
						d6 = 1.0 / (x6 * x6 + y6 * y6);
						d7 = 1.0 / (x7 * x7 + y7 * y7);       
						//dist2 = 1.0 / (distX * distX + distY * distY);
                                                
						fx0 -= x0 * d0;
						fx1 -= x1 * d1;
						fx2 -= x2 * d2;
                                                fx3 -= x3 * d3;
						fx4 -= x4 * d4;
                                                fx5 -= x5 * d5;
                                                fx6 -= x6 * d6;
                                                fx7 -= x7 * d7;
						fy0 -= y0 * d0;
						fy1 -= y1 * d1;
						fy2 -= y2 * d2;
                                                fy3 -= y3 * d3;
						fy4 -= y4 * d4;
                                                fy5 -= y5 * d5;
                                                fy6 -= y6 * d6;
                                                fy7 -= y7 * d7;
						//fx += distX * dist2;
                                                //fy += distY * dist2;
                                        }
					for(INDEXTYPE j = i+1; j < i + 8; j++){
						x0 = blasX[j] - x0;
						y0 = blasY[j] - y0;
						d0 = 1.0 / (x0 * x0 + y0 * y0);	
						fx0 -= x0 * d0;
						fy0 -= y0 * d0;				
					}
					x1 = blasX[i] - x1;
					y1 = blasX[i] - y1;
					d1 = 1.0 / (x1 * x1 + y1 * y1);
					fx1 -= x1 * d1;
					fy1 -= y1 * d1;
					for(INDEXTYPE j = i+2; j < i + 8; j++){
                                                x1 = blasX[j] - x1;
                                                y1 = blasY[j] - y1;
                                                d1 = 1.0 / (x1 * x1 + y1 * y1);
                                                fx1 -= x1 * d1;
                                                fy1 -= y1 * d1;
                                        }
					for(INDEXTYPE j = i; j < i + 2; j++){
                                                x2 = blasX[j] - x2;
                                                y2 = blasY[j] - y2;
                                                d2 = 1.0 / (x2 * x2 + y2 * y2);
                                                fx2 -= x2 * d2;
                                                fy2 -= y2 * d2;
                                        }
					for(INDEXTYPE j = i+3; j < i + 8; j++){
                                                x2 = blasX[j] - x2;
                                                y2 = blasY[j] - y2;
                                                d2 = d0 = 1.0 / (x2 * x2 + y2 * y2);
                                                fx2 -= x2 * d2;
                                                fy2 -= y2 * d2;
                                        }
					//////////////////
					for(INDEXTYPE j = i; j < i + 3; j++){
                                                x3 = blasX[j] - x3;
                                                y3 = blasY[j] - y3;
                                                d3 = 1.0 / (x3 * x3 + y3 * y3);
                                                fx3 -= x3 * d3;
                                                fy3 -= y3 * d3;
                                        }
					for(INDEXTYPE j = i+4; j < i + 8; j++){
                                                x3 = blasX[j] - x3;
                                                y3 = blasY[j] - y3;
                                                d3 = 1.0 / (x3 * x3 + y3 * y3);
                                                fx3 -= x3 * d3;
                                                fy3 -= y3 * d3;
                                        }
					////////
					for(INDEXTYPE j = i; j < i + 4; j++){
                                                x4 = blasX[j] - x4;
                                                y4 = blasY[j] - y4;
                                                d4 = 1.0 / (x4 * x4 + y4 * y4);
                                                fx4 -= x4 * d4;
                                                fy4 -= y4 * d4;
                                        }
					for(INDEXTYPE j = i+5; j < i + 8; j++){
                                                x4 = blasX[j] - x4;
                                                y4 = blasY[j] - y4;
                                                d4 = 1.0 / (x4 * x4 + y4 * y4);
                                                fx4 -= x4 * d4;
                                                fy4 -= y4 * d4;
                                        }
					/////
					for(INDEXTYPE j = i; j < i + 5; j++){
                                                x5 = blasX[j] - x5;
                                                y5 = blasY[j] - y5;
                                                d5 = 1.0 / (x5 * x5 + y5 * y5);
                                                fx5 -= x5 * d5;
                                                fy5 -= y5 * d5;
                                        }
					for(INDEXTYPE j = i+6; j < i + 8; j++){
                                                x5 = blasX[j] - x5;
                                                y5 = blasY[j] - y5;
                                                d5 = 1.0 / (x5 * x5 + y5 * y5);
                                                fx5 -= x5 * d5;
                                                fy5 -= y5 * d5;
                                        }
			
					//////
					for(INDEXTYPE j = i; j < i + 6; j++){
                                                x6 = blasX[j] - x6;
                                                y6 = blasY[j] - y6;
                                                d6 = 1.0 / (x6 * x6 + y6 * y6);
                                                fx6 -= x6 * d6;
                                                fy6 -= y6 * d6;
                                        }
					x6 = blasX[i+7] - x6;
                                        y6 = blasX[i+7] - y6;
                                        d6 = 1.0 / (x6 * x6 + y6 * y6);
                                        fx6 -= x6 * d6;
                                        fy6 -= y6 * d1;
					////
					for(INDEXTYPE j = i; j < i + 7; j++){
                                                x7 = blasX[j] - x7;
                                                y7 = blasY[j] - y7;
                                                d7 = 1.0 / (x7 * x7 + y7 * y7);
                                                fx7 -= x7 * d7;
                                                fy7 -= y7 * d7;
                                        }
					////////////
                                        for(INDEXTYPE j = i+8; j < graph.rows; j += 1){
                                                VALUETYPE vx = blasX[j];
						VALUETYPE vy = blasY[j];
						x0 = vx - x0;
                                                x1 = vx - x1;
                                                x2 = vx - x2;
                                                x3 = vx - x3;
                                                x4 = vx - x4;
                                                x5 = vx - x5;
                                                x6 = vx - x6;
                                                x7 = vx - x7;
                                                y0 = vy - y0;
                                                y1 = vy - y1;
                                                y2 = vy - y2;
                                                y3 = vy - y3;
                                                y4 = vy - y4;
                                                y5 = vy - y5;
                                                y6 = vy - y6;
                                                y7 = vy - y7;
						//distX = blasX[j] - blasX[i];
                                                //distY = blasY[j] - blasY[i];
                                                
						d0 = 1.0 / (x0 * x0 + y0 * y0);
                                                d1 = 1.0 / (x1 * x1 + y1 * y1);
                                                d2 = 1.0 / (x2 * x2 + y2 * y2);
                                                d3 = 1.0 / (x3 * x3 + y3 * y3);
                                                d4 = 1.0 / (x4 * x4 + y4 * y4);
                                                d5 = 1.0 / (x5 * x5 + y5 * y5);
                                                d6 = 1.0 / (x6 * x6 + y6 * y6);
                                                d7 = 1.0 / (x7 * x7 + y7 * y7);
						//dist2 = 1.0 / (distX * distX + distY * distY);
                                                
						fx0 -= x0 * d0;
                                                fx1 -= x1 * d1;
                                                fx2 -= x2 * d2;
                                                fx3 -= x3 * d3;
                                                fx4 -= x4 * d4;
                                                fx5 -= x5 * d5;
                                                fx6 -= x6 * d6;
                                                fx7 -= x7 * d7;
                                                fy0 -= y0 * d0;
                                                fy1 -= y1 * d1;
                                                fy2 -= y2 * d2;
                                                fy3 -= y3 * d3;
                                                fy4 -= y4 * d4;
                                                fy5 -= y5 * d5;
                                                fy6 -= y6 * d6;
                                                fy7 -= y7 * d7;
						//fx += distX * dist2;
                                                //fy += distY * dist2;
                                        }
					pb_X[ind] = fx0;
					pb_X[ind+1] = fx1;
					pb_X[ind+2] = fx2;
                                        pb_X[ind+3] = fx3;
					pb_X[ind+4] = fx4;
                                        pb_X[ind+5] = fx5;
                                        pb_X[ind+6] = fx6;
                                        pb_X[ind+7] = fx7;
					
					pb_Y[ind] = fy0;
					pb_Y[ind+1] = fy1;
					pb_Y[ind+2] = fy2;
                                        pb_Y[ind+3] = fy3;
					pb_Y[ind+4] = fy4;
                                        pb_Y[ind+5] = fy5;
                                        pb_Y[ind+6] = fy6;
                                        pb_Y[ind+7] = fy7;	
				}
				/*printf("After:\n");
                                for(INDEXTYPE i = b * BATCHSIZE; i < (b + 1) * BATCHSIZE; i++){
                                        printf("%d = %lf,", i, pb_X[i-b*BATCHSIZE]);
                                }
				*/
				for(INDEXTYPE i = b * BATCHSIZE; i < (b + 1) * BATCHSIZE; i += 1){
                                        int ind = i-b*BATCHSIZE;
                                        VALUETYPE d;
					d = (pb_X[ind] * pb_X[ind] + pb_Y[ind] * pb_Y[ind]);
					ENERGY += d ;
					d = STEP / sqrt(d);
					blasX[i] += pb_X[ind] * d;
                                        blasY[i] += pb_Y[ind] * d;
					pb_X[ind] = 0;
					pb_Y[ind] = 0;    
                                }
                        }
			INDEXTYPE cleanup = (graph.rows/BATCHSIZE) * BATCHSIZE;
			#pragma omp parallel for schedule(dynamic) 
			for(INDEXTYPE i = cleanup; i < graph.rows; i += 1){
                                INDEXTYPE ind = i- cleanup;
                                VALUETYPE fx = 0, fy = 0, distX, distY, dist, dist2;
                                for(INDEXTYPE j = graph.rowptr[i]; j < graph.rowptr[i+1]; j += 1){
                                        int v = graph.colids[j];
                                        distX = blasX[v] - blasX[i];
                                        distY = blasY[v] - blasY[i];
                                        dist = (distX * distX + distY * distY);
                                        dist = sqrt(dist) + 1.0 / dist;
                                        pb_X[ind] += distX * dist;
                                        pb_Y[ind] += distY * dist;
                                }
                                for(INDEXTYPE j = 0; j < i; j += 1){
                                        distX = blasX[j] - blasX[i];
                                        distY = blasY[j] - blasY[i];
                                        dist2 = 1.0 / (distX * distX + distY * distY);
                                        fx += distX * dist2;
                                        fy += distY * dist2;
                                }
                                for(INDEXTYPE j = i+1; j < graph.rows; j += 1){
                                        distX = blasX[j] - blasX[i];
                                        distY = blasY[j] - blasY[i];
                                        dist2 = 1.0 / (distX * distX + distY * distY);
                                        fx += distX * dist2;
                                        fy += distY * dist2;
                                }
                                pb_X[ind] = pb_X[ind] - fx;
                                pb_Y[ind] = pb_Y[ind] - fy;
                        }	
			for(INDEXTYPE i = cleanup; i < graph.rows; i += 1){
                                int ind = i-cleanup;
                                double dist = (pb_X[ind]*pb_X[ind] + pb_Y[ind]*pb_Y[ind]);
                                ENERGY += dist;
				dist = STEP / sqrt(dist);
				blasX[i] += pb_X[ind] * dist;
                                blasY[i] += pb_Y[ind] * dist;
				pb_X[ind] = pb_Y[ind] = 0;
                        }
                        STEP = STEP * 0.999;
                        LOOP++;
                }
                end = omp_get_wtime();
            #if NO_FWRITE == 0
                cout << "Efficientunroll Minibatch Size:" << BATCHSIZE  << endl;
                cout << "Efficientunroll Minbatch Energy:" << ENERGY << endl;
                cout << "Efficientunroll Minibatch Parallel Wall time required:" << end - start << endl;
                writeToFileEFF("EFFUR"+ to_string(BATCHSIZE)+"PARAOUT" + to_string(LOOP));
            #endif
                result.push_back(ENERGY);
                result.push_back(end - start);
                return result;
        }
#else 
/*
 * ************************ N-DIM : Unroll and Jam Impl ******************************
 * NOTE: define for appropriate unroll factor 
 *
 * NOTE: NOTE: 
 *             Unroll = 8 .... vectorized by compiler... 
 *             So, implicit unrolling = 8 * 8 = 64
 *             So, N-dim threading .... only 256 / 64 = 4 threads is active!!!
 * When Unroll = 1  // compiler vectorized the code so, implicit unrolling 8
 *    Active iteration = 256 / 8 = 32 
 *    18 threads, per share 1+ 
 *    NOTE: that's why when Unroll = 2, got less than 1 iteration !!!!! 
 */
   //#define UR4_MEMOPT 1 
   #define UR8 1 
   #define DEBUG 0

   #ifdef UR8 
   vector<VALUETYPE> newalgo::EfficientVersionUnRoll(INDEXTYPE ITERATIONS, 
         INDEXTYPE NUMOFTHREADS, INDEXTYPE BATCHSIZE)
   {
      INDEXTYPE LOOP = 0;
      VALUETYPE start, end, ENERGY, ENERGY0, *pb_X, *pb_Y;
      VALUETYPE STEP = 1.0;
      
      vector<VALUETYPE> result;
      pb_X = static_cast<VALUETYPE *> (::operator new (sizeof(VALUETYPE[BATCHSIZE])));
      pb_Y = static_cast<VALUETYPE *> (::operator new (sizeof(VALUETYPE[BATCHSIZE])));
      ENERGY0 = ENERGY = numeric_limits<VALUETYPE>::max();
     
#if 0
      cout << "Rows = " << graph.rows << endl;
      exit(1);
#endif

      omp_set_num_threads(NUMOFTHREADS);
/*
 *    time included initDFS 
 */
      start = omp_get_wtime();

      initDFS();
      
      while(LOOP < ITERATIONS)
      {
         ENERGY0 = ENERGY;
         ENERGY = 0;
         for(int i = 0; i < BATCHSIZE; i++)
         {
            pb_X[i] = pb_Y[i] = 0;
         }
	
         
	 for(INDEXTYPE b = 0; b < (graph.rows / BATCHSIZE); b += 1)
         {

/*
 *          NOTE: no cleanup for unroll as long as BATCHZIE is multiple of unroll
 *          factor 
 */
	    #pragma omp parallel for schedule(static)	
	    for(INDEXTYPE i = b * BATCHSIZE; i < (b + 1) * BATCHSIZE; i += 8)
            {
	       int ind = i-b*BATCHSIZE;

               VALUETYPE distX, distY, dist; 
               VALUETYPE fx0, fx1, fx2, fx3, fx4, fx5, fx6, fx7;
               VALUETYPE fy0, fy1, fy2, fy3, fy4, fy5, fy6, fy7;
               
               
               VALUETYPE x0, x1, x2, x3, x4, x5, x6, x7;
               VALUETYPE y0, y1, y2, y3, y4, y5, y6, y7;
               VALUETYPE d0, d1, d2, d3, d4, d5, d6, d7;
					
	       x0 = blasX[i];
	       x1 = blasX[i+1];
	       x2 = blasX[i+2];
	       x3 = blasX[i+3];
	       x4 = blasX[i+4];
	       x5 = blasX[i+5];
	       x6 = blasX[i+6];
	       x7 = blasX[i+7];

	       y0 = blasY[i];
	       y1 = blasY[i+1];	
	       y2 = blasY[i+2];
	       y3 = blasY[i+3];
	       y4 = blasY[i+4];
	       y5 = blasY[i+5];
	       y6 = blasY[i+6];
	       y7 = blasY[i+7];
					
	       fx0 = fx1 = fx2 = fx3 = fx4 = fx5 = fx6 = fx7 = 0;
	       fy0 = fy1 = fy2 = fy3 = fy4 = fy5 = fy6 = fy7 = 0;		
		
/*
 *              j is up to i... lower up case  
 */
               for(INDEXTYPE j = 0; j < i; j += 1)
               {
                  VALUETYPE dx0, dx1, dx2, dx3, dx4, dx5, dx6, dx7;
                  VALUETYPE dy0, dy1, dy2, dy3, dy4, dy5, dy6, dy7;
                  VALUETYPE xj = blasX[j];
                  VALUETYPE yj = blasY[j];
		  
                  dx0 = xj - x0;
		  dx1 = xj - x1;
		  dx2 = xj - x2;
                  dx3 = xj - x3;
		  dx4 = xj - x4;
                  dx5 = xj - x5;
                  dx6 = xj - x6;
                  dx7 = xj - x7;
	          
                  dy0 = yj - y0;
		  dy1 = yj - y1;
		  dy2 = yj - y2;
                  dy3 = yj - y3;
		  dy4 = yj - y4;
                  dy5 = yj - y5;
                  dy6 = yj - y6;
                  dy7 = yj - y7; 	       
		
                  //distX = blasX[j] - blasX[i];
                  //distY = blasY[j] - blasY[i];
                                   	
	          d0 = 1.0 / (dx0 * dx0 + dy0 * dy0);
		  d1 = 1.0 / (dx1 * dx1 + dy1 * dy1);
		  d2 = 1.0 / (dx2 * dx2 + dy2 * dy2);
		  d3 = 1.0 / (dx3 * dx3 + dy3 * dy3);
		  d4 = 1.0 / (dx4 * dx4 + dy4 * dy4);
		  d5 = 1.0 / (dx5 * dx5 + dy5 * dy5);
		  d6 = 1.0 / (dx6 * dx6 + dy6 * dy6);
		  d7 = 1.0 / (dx7 * dx7 + dy7 * dy7);       
		
                  //dist2 = 1.0 / (distX * distX + distY * distY);
                                            
		  fx0 += dx0 * d0;
		  fx1 += dx1 * d1;
		  fx2 += dx2 * d2;
                  fx3 += dx3 * d3;
		  fx4 += dx4 * d4;
                  fx5 += dx5 * d5;
                  fx6 += dx6 * d6;
                  fx7 += dx7 * d7;
		
                  fy0 += dy0 * d0;
		  fy1 += dy1 * d1;
		  fy2 += dy2 * d2;
                  fy3 += dy3 * d3;
		  fy4 += dy4 * d4;
                  fy5 += dy5 * d5;
                  fy6 += dy6 * d6;
                  fy7 += dy7 * d7;
		 
                  //fx += distX * dist2;
                  //fy += distY * dist2;
               }		
/*
 *             location where we need to skip some points i==j
 *             NOTE: its UR iteration, no need to optimize 
 *             Hopefully compiler will unroll and get rid of the conditions
 */
               for (INDEXTYPE j = i; j < i+8; j++) 
               {
                  VALUETYPE xj = blasX[j];
                  VALUETYPE yj = blasY[j];
		  
                  if (j != i)
                  {
                     VALUETYPE dx0, dy0;
                     dx0 = xj - x0;
                     dy0 = yj - y0;
	             d0 = 1.0 / (dx0 * dx0 + dy0 * dy0);
		     fx0 += dx0 * d0;
                     fy0 += dy0 * d0;
                  }
                  if ( j != i+1 )
                  {
                     VALUETYPE dx1, dy1;
                     dx1 = xj - x1;
                     dy1 = yj - y1;
	             d1 = 1.0 / (dx1 * dx1 + dy1 * dy1);
		     fx1 += dx1 * d1;
                     fy1 += dy1 * d1;
                  }
                  if ( j != i+2 )
                  {
                     VALUETYPE dx2, dy2;
                     dx2 = xj - x2;
                     dy2 = yj - y2;
	             d2 = 1.0 / (dx2 * dx2 + dy2 * dy2);
		     fx2 += dx2 * d2;
                     fy2 += dy2 * d2;
                  }
                  if ( j != i+3 )
                  {
                     VALUETYPE dx3, dy3;
                     dx3 = xj - x3;
                     dy3 = yj - y3;
	             d3 = 1.0 / (dx3 * dx3 + dy3 * dy3);
		     fx3 += dx3 * d3;
                     fy3 += dy3 * d3;
                  }
                  if ( j != i+4 )
                  {
                     VALUETYPE dx4, dy4;
                     dx4 = xj - x4;
                     dy4 = yj - y4;
	             d4 = 1.0 / (dx4 * dx4 + dy4 * dy4);
		     fx4 += dx4 * d4;
                     fy4 += dy4 * d4;
                  }
                  if ( j != i+5 )
                  {
                     VALUETYPE dx5, dy5;
                     dx5 = xj - x5;
                     dy5 = yj - y5;
	             d5 = 1.0 / (dx5 * dx5 + dy5 * dy5);
		     fx5 += dx5 * d5;
                     fy5 += dy5 * d5;
                  }
                  if ( j != i+6 )
                  {
                     VALUETYPE dx6, dy6;
                     dx6 = xj - x6;
                     dy6 = yj - y6;
	             d6 = 1.0 / (dx6 * dx6 + dy6 * dy6);
		     fx6 += dx6 * d6;
                     fy6 += dy6 * d6;
                  }
                  if ( j != i+7 )
                  {
                     VALUETYPE dx7, dy7;
                     dx7 = xj - x7;
                     dy7 = yj - y7;
	             d7 = 1.0 / (dx7 * dx7 + dy7 * dy7);
		     fx7 += dx7 * d7;
                     fy7 += dy7 * d7;
                  }
               }
/*
 *             Upper part  
 *
 */ 
               for(INDEXTYPE j = i+8; j < graph.rows; j += 1)
               {
                  VALUETYPE dx0, dx1, dx2, dx3, dx4, dx5, dx6, dx7;
                  VALUETYPE dy0, dy1, dy2, dy3, dy4, dy5, dy6, dy7;
                  VALUETYPE xj = blasX[j];
                  VALUETYPE yj = blasY[j];
		  
                  dx0 = xj - x0;
		  dx1 = xj - x1;
		  dx2 = xj - x2;
                  dx3 = xj - x3;
		  dx4 = xj - x4;
                  dx5 = xj - x5;
                  dx6 = xj - x6;
                  dx7 = xj - x7;
	          
                  dy0 = yj - y0;
		  dy1 = yj - y1;
		  dy2 = yj - y2;
                  dy3 = yj - y3;
		  dy4 = yj - y4;
                  dy5 = yj - y5;
                  dy6 = yj - y6;
                  dy7 = yj - y7; 	       
		
                  //distX = blasX[j] - blasX[i];
                  //distY = blasY[j] - blasY[i];
                                   	
	          d0 = 1.0 / (dx0 * dx0 + dy0 * dy0);
		  d1 = 1.0 / (dx1 * dx1 + dy1 * dy1);
		  d2 = 1.0 / (dx2 * dx2 + dy2 * dy2);
		  d3 = 1.0 / (dx3 * dx3 + dy3 * dy3);
		  d4 = 1.0 / (dx4 * dx4 + dy4 * dy4);
		  d5 = 1.0 / (dx5 * dx5 + dy5 * dy5);
		  d6 = 1.0 / (dx6 * dx6 + dy6 * dy6);
		  d7 = 1.0 / (dx7 * dx7 + dy7 * dy7);       
		
                  //dist2 = 1.0 / (distX * distX + distY * distY);
                                            
		  fx0 += dx0 * d0;
		  fx1 += dx1 * d1;
		  fx2 += dx2 * d2;
                  fx3 += dx3 * d3;
		  fx4 += dx4 * d4;
                  fx5 += dx5 * d5;
                  fx6 += dx6 * d6;
                  fx7 += dx7 * d7;
		
                  fy0 += dy0 * d0;
		  fy1 += dy1 * d1;
		  fy2 += dy2 * d2;
                  fy3 += dy3 * d3;
		  fy4 += dy4 * d4;
                  fy5 += dy5 * d5;
                  fy6 += dy6 * d6;
                  fy7 += dy7 * d7;
		 
                  //fx += distX * dist2;
                  //fy += distY * dist2;
               }
               
               
               // connected nodes 
               VALUETYPE px0=0, px1=0, px2=0, px3=0, px4=0, px5=0, px6=0, px7=0;
               VALUETYPE py0=0, py1=0, py2=0, py3=0, py4=0, py5=0, py6=0, py7=0;
            
            #if DEBUG == 1
                  cout << "Loop = "<< LOOP <<" i = " << i << 
                     ": connected nodes = " << 
                     graph.rowptr[i+1] - graph.rowptr[i] << endl;
            #endif
               
               for(INDEXTYPE j = graph.rowptr[i]; j < graph.rowptr[i+1]; j += 1)
               {
                  int v = graph.colids[j];
                  distX = blasX[v] - x0;
                  distY = blasY[v] - y0;
                  dist = (distX * distX + distY * distY);
                  dist = sqrt(dist) + 1.0 / dist;

                  px0 += distX * dist;
                  py0 += distY * dist;
               }
            #if DEBUG == 1
                  cout << "i = " << i+1 << ": connected nodes = " << 
                     graph.rowptr[i+2] - graph.rowptr[i+1] << endl;
            #endif
					
	       for(INDEXTYPE j = graph.rowptr[i+1]; j < graph.rowptr[i+1+1]; j += 1)
               {
                  int v = graph.colids[j];
                  
                  distX = blasX[v] - x1;
                  distY = blasY[v] - y1;
 
                  dist = (distX * distX + distY * distY);
                  dist = sqrt(dist) + 1.0 / dist;

                  px1 += distX * dist;
                  py1 += distY * dist;
               }
				
            #if DEBUG == 1
                  cout << "i = " << i+2 << ": connected nodes = " << 
                     graph.rowptr[i+3] - graph.rowptr[i+2] << endl;
            #endif
	       for(INDEXTYPE j = graph.rowptr[i+2]; j < graph.rowptr[i+1+2]; j += 1)
               {
                  int v = graph.colids[j];
                  distX = blasX[v] - x2;
                  distY = blasY[v] - y2;

                  dist = (distX * distX + distY * distY);
                  dist = sqrt(dist) + 1.0 / dist;

                  px2 += distX * dist;
                  py2 += distY * dist;
               }

            #if DEBUG == 1
                  cout << "i = " << i+3 << ": connected nodes = " << 
                     graph.rowptr[i+4] - graph.rowptr[i+3] << endl;
            #endif
	       for(INDEXTYPE j = graph.rowptr[i+3]; j < graph.rowptr[i+1+3]; j += 1)
               {
                  int v = graph.colids[j];
                  distX = blasX[v] - x3;
                  distY = blasY[v] - y3;

                  dist = (distX * distX + distY * distY);
                  dist = sqrt(dist) + 1.0 / dist;

                  px3 += distX * dist;
                  py3 += distY * dist;                         
               }

            #if DEBUG == 1
                  cout << "i = " << i+4 << ": connected nodes = " << 
                     graph.rowptr[i+5] - graph.rowptr[i+4] << endl;
            #endif
	       for(INDEXTYPE j = graph.rowptr[i+4]; j < graph.rowptr[i+1+4]; j += 1)
               {
                  int v = graph.colids[j];
                  distX = blasX[v] - x4;
                  distY = blasY[v] - y4;

                  dist = (distX * distX + distY * distY);
                  dist = sqrt(dist) + 1.0 / dist;

                  px4 += distX * dist;
                  py4 += distY * dist;
               }
            #if DEBUG == 1
                  cout << "i = " << i+5 << ": connected nodes = " << 
                     graph.rowptr[i+6] - graph.rowptr[i+5] << endl;
            #endif
	       for(INDEXTYPE j = graph.rowptr[i+5]; j < graph.rowptr[i+1+5]; j += 1)
               {
                  int v = graph.colids[j];
                  distX = blasX[v] - x5;
                  distY = blasY[v] - y5;

                  dist = (distX * distX + distY * distY);
                  dist = sqrt(dist) + 1.0 / dist;

                  px5 += distX * dist;
                  py5 += distY * dist;
               }
            #if DEBUG == 1
                  cout << "i = " << i+6 << ": connected nodes = " << 
                     graph.rowptr[i+7] - graph.rowptr[i+6] << endl;
            #endif
	       for(INDEXTYPE j = graph.rowptr[i+6]; j < graph.rowptr[i+1+6]; j += 1)
               {
                  int v = graph.colids[j];
                  distX = blasX[v] - x6;
                  distY = blasY[v] - y6;

                  dist = (distX * distX + distY * distY);
                  dist = sqrt(dist) + 1.0 / dist;

                  px6 += distX * dist;
                  py6 += distY * dist;                         
               }
            #if DEBUG == 1
                  cout << "i = " << i+7 << ": connected nodes = " << 
                     graph.rowptr[i+8] - graph.rowptr[i+7] << endl;
            #endif
	       for(INDEXTYPE j = graph.rowptr[i+7]; j < graph.rowptr[i+1+7]; j += 1)
               {
                  int v = graph.colids[j];
                  distX = blasX[v] - x7;
                  distY = blasY[v] - y7;
   
                  dist = (distX * distX + distY * distY);
                  dist = sqrt(dist) + 1.0 / dist;

                  px7 += distX * dist;
                  py7 += distY * dist;
                }
	       
               pb_X[ind] = px0 - fx0;
	       pb_X[ind+1] = px1 - fx1;
	       pb_X[ind+2] = px2 - fx2;
               pb_X[ind+3] = px3 - fx3;
	       pb_X[ind+4] = px4 - fx4;
               pb_X[ind+5] = px5 - fx5;
               pb_X[ind+6] = px6 - fx6;
               pb_X[ind+7] = px7- fx7;
					
	       pb_Y[ind] = py0 - fy0;
	       pb_Y[ind+1] = py1 - fy1;
	       pb_Y[ind+2] = py2 - fy2;
               pb_Y[ind+3] = py3 - fy3;
	       pb_Y[ind+4] = py4 - fy4;
               pb_Y[ind+5] = py5 - fy5;
               pb_Y[ind+6] = py5 - fy6;
               pb_Y[ind+7] = py6 - fy7;	
	    }
	    /*printf("After:\n");
            for(INDEXTYPE i = b * BATCHSIZE; i < (b + 1) * BATCHSIZE; i++)
            {
               printf("%d = %lf,", i, pb_X[i-b*BATCHSIZE]);
            }
	    */
	    for(INDEXTYPE i = b * BATCHSIZE; i < (b + 1) * BATCHSIZE; i += 1)
            {
               int ind = i-b*BATCHSIZE;
               VALUETYPE d;
	       d = (pb_X[ind] * pb_X[ind] + pb_Y[ind] * pb_Y[ind]);
	       ENERGY += d ;
	       
               d = STEP / sqrt(d);
	       blasX[i] += pb_X[ind] * d;
               blasY[i] += pb_Y[ind] * d;
	       
               pb_X[ind] = 0;
	       pb_Y[ind] = 0;    
            }
         }
/*
 *    cleanup 
 */
			INDEXTYPE cleanup = (graph.rows/BATCHSIZE) * BATCHSIZE;
			#pragma omp parallel for schedule(dynamic) 
			for(INDEXTYPE i = cleanup; i < graph.rows; i += 1){
                                INDEXTYPE ind = i- cleanup;
                                VALUETYPE fx = 0, fy = 0, distX, distY, dist, dist2;
                                for(INDEXTYPE j = graph.rowptr[i]; j < graph.rowptr[i+1]; j += 1){
                                        int v = graph.colids[j];
                                        distX = blasX[v] - blasX[i];
                                        distY = blasY[v] - blasY[i];
                                        dist = (distX * distX + distY * distY);
                                        dist = sqrt(dist) + 1.0 / dist;
                                        pb_X[ind] += distX * dist;
                                        pb_Y[ind] += distY * dist;
                                }
                                for(INDEXTYPE j = 0; j < i; j += 1){
                                        distX = blasX[j] - blasX[i];
                                        distY = blasY[j] - blasY[i];
                                        dist2 = 1.0 / (distX * distX + distY * distY);
                                        fx += distX * dist2;
                                        fy += distY * dist2;
                                }
                                for(INDEXTYPE j = i+1; j < graph.rows; j += 1){
                                        distX = blasX[j] - blasX[i];
                                        distY = blasY[j] - blasY[i];
                                        dist2 = 1.0 / (distX * distX + distY * distY);
                                        fx += distX * dist2;
                                        fy += distY * dist2;
                                }
                                pb_X[ind] = pb_X[ind] - fx;
                                pb_Y[ind] = pb_Y[ind] - fy;
                        }	
			for(INDEXTYPE i = cleanup; i < graph.rows; i += 1){
                                int ind = i-cleanup;
                                double dist = (pb_X[ind]*pb_X[ind] + pb_Y[ind]*pb_Y[ind]);
                                ENERGY += dist;
				dist = STEP / sqrt(dist);
				blasX[i] += pb_X[ind] * dist;
                                blasY[i] += pb_Y[ind] * dist;
				pb_X[ind] = pb_Y[ind] = 0;
                        }
                        STEP = STEP * 0.999;
                        LOOP++;
                }
                end = omp_get_wtime();
            #if NO_FWRITE == 0
                cout << "Efficientunroll Minibatch Size:" << BATCHSIZE  << endl;
                cout << "Efficientunroll Minbatch Energy:" << ENERGY << endl;
                cout << "Efficientunroll Minibatch Parallel Wall time required:" << end - start << endl;
                writeToFileEFF("EFFUR"+ to_string(BATCHSIZE)+"PARAOUT" + to_string(LOOP));
            #endif
                result.push_back(ENERGY);
                result.push_back(end - start);
                return result;
        }
   #elif defined(UR4)
/*
 * **************************** UNROLL FACTOR = 4 ************************ 
 */
   vector<VALUETYPE> newalgo::EfficientVersionUnRoll(INDEXTYPE ITERATIONS, 
         INDEXTYPE NUMOFTHREADS, INDEXTYPE BATCHSIZE)
   {
      INDEXTYPE LOOP = 0;
      VALUETYPE start, end, ENERGY, ENERGY0, *pb_X, *pb_Y;
      VALUETYPE STEP = 1.0;
      
      vector<VALUETYPE> result;
      pb_X = static_cast<VALUETYPE *> (::operator new (sizeof(VALUETYPE[BATCHSIZE])));
      pb_Y = static_cast<VALUETYPE *> (::operator new (sizeof(VALUETYPE[BATCHSIZE])));
      ENERGY0 = ENERGY = numeric_limits<VALUETYPE>::max();
     
#if 0
      cout << "Rows = " << graph.rows << endl;
      exit(1);
#endif

      omp_set_num_threads(NUMOFTHREADS);
/*
 *    time included initDFS 
 */
      start = omp_get_wtime();

      initDFS();
      
      while(LOOP < ITERATIONS)
      {
         ENERGY0 = ENERGY;
         ENERGY = 0;
         for(int i = 0; i < BATCHSIZE; i++)
         {
            pb_X[i] = pb_Y[i] = 0;
         }
	
         
	 for(INDEXTYPE b = 0; b < (graph.rows / BATCHSIZE); b += 1)
         {

/*
 *          NOTE: no cleanup for unroll as lon as BATCHZIE is multiple of unroll
 *          factor 
 */
	    #pragma omp parallel for schedule(static)	
	    for(INDEXTYPE i = b * BATCHSIZE; i < (b + 1) * BATCHSIZE; i += 4)
            {
	       int ind = i-b*BATCHSIZE;

               VALUETYPE distX, distY, dist; 
               VALUETYPE fx0, fx1, fx2, fx3;
               VALUETYPE fy0, fy1, fy2, fy3;
               
               
               VALUETYPE x0, x1, x2, x3;
               VALUETYPE y0, y1, y2, y3;
               VALUETYPE d0, d1, d2, d3;
					
	       x0 = blasX[i];
	       x1 = blasX[i+1];
	       x2 = blasX[i+2];
	       x3 = blasX[i+3];

	       y0 = blasY[i];
	       y1 = blasY[i+1];	
	       y2 = blasY[i+2];
	       y3 = blasY[i+3];
					
	       fx0 = fx1 = fx2 = fx3 = 0;
	       fy0 = fy1 = fy2 = fy3 = 0;		
		
/*
 *              j is up to i... lower up case  
 */
               for(INDEXTYPE j = 0; j < i; j += 1)
               {
                  VALUETYPE dx0, dx1, dx2, dx3;
                  VALUETYPE dy0, dy1, dy2, dy3;
                  VALUETYPE xj = blasX[j];
                  VALUETYPE yj = blasY[j];
		  
                  dx0 = xj - x0;
		  dx1 = xj - x1;
		  dx2 = xj - x2;
                  dx3 = xj - x3;
	          
                  dy0 = yj - y0;
		  dy1 = yj - y1;
		  dy2 = yj - y2;
                  dy3 = yj - y3;
		
                  //distX = blasX[j] - blasX[i];
                  //distY = blasY[j] - blasY[i];
                                   	
	          d0 = 1.0 / (dx0 * dx0 + dy0 * dy0);
		  d1 = 1.0 / (dx1 * dx1 + dy1 * dy1);
		  d2 = 1.0 / (dx2 * dx2 + dy2 * dy2);
		  d3 = 1.0 / (dx3 * dx3 + dy3 * dy3);
		
                  //dist2 = 1.0 / (distX * distX + distY * distY);
                                            
		  fx0 += dx0 * d0;
		  fx1 += dx1 * d1;
		  fx2 += dx2 * d2;
                  fx3 += dx3 * d3;
		
                  fy0 += dy0 * d0;
		  fy1 += dy1 * d1;
		  fy2 += dy2 * d2;
                  fy3 += dy3 * d3;
		 
                  //fx += distX * dist2;
                  //fy += distY * dist2;
               }		
/*
 *             location where we need to skip some points i==j
 *             NOTE: its UR iteration, no need to optimize 
 *             Hopefully compiler will unroll and get rid of the conditions
 */
               for (INDEXTYPE j = i; j < i+4; j++) 
               {
                  VALUETYPE xj = blasX[j];
                  VALUETYPE yj = blasY[j];
		  
                  if (j != i)
                  {
                     VALUETYPE dx0, dy0;
                     dx0 = xj - x0;
                     dy0 = yj - y0;
	             d0 = 1.0 / (dx0 * dx0 + dy0 * dy0);
		     fx0 += dx0 * d0;
                     fy0 += dy0 * d0;
                  }
                  if ( j != i+1 )
                  {
                     VALUETYPE dx1, dy1;
                     dx1 = xj - x1;
                     dy1 = yj - y1;
	             d1 = 1.0 / (dx1 * dx1 + dy1 * dy1);
		     fx1 += dx1 * d1;
                     fy1 += dy1 * d1;
                  }
                  if ( j != i+2 )
                  {
                     VALUETYPE dx2, dy2;
                     dx2 = xj - x2;
                     dy2 = yj - y2;
	             d2 = 1.0 / (dx2 * dx2 + dy2 * dy2);
		     fx2 += dx2 * d2;
                     fy2 += dy2 * d2;
                  }
                  if ( j != i+3 )
                  {
                     VALUETYPE dx3, dy3;
                     dx3 = xj - x3;
                     dy3 = yj - y3;
	             d3 = 1.0 / (dx3 * dx3 + dy3 * dy3);
		     fx3 += dx3 * d3;
                     fy3 += dy3 * d3;
                  }
               }
/*
 *             Upper part  
 *
 */ 
               for(INDEXTYPE j = i+4; j < graph.rows; j += 1)
               {
                  VALUETYPE dx0, dx1, dx2, dx3;
                  VALUETYPE dy0, dy1, dy2, dy3;
                  VALUETYPE xj = blasX[j];
                  VALUETYPE yj = blasY[j];
		  
                  dx0 = xj - x0;
		  dx1 = xj - x1;
		  dx2 = xj - x2;
                  dx3 = xj - x3;
	          
                  dy0 = yj - y0;
		  dy1 = yj - y1;
		  dy2 = yj - y2;
                  dy3 = yj - y3;
		
                  //distX = blasX[j] - blasX[i];
                  //distY = blasY[j] - blasY[i];
                                   	
	          d0 = 1.0 / (dx0 * dx0 + dy0 * dy0);
		  d1 = 1.0 / (dx1 * dx1 + dy1 * dy1);
		  d2 = 1.0 / (dx2 * dx2 + dy2 * dy2);
		  d3 = 1.0 / (dx3 * dx3 + dy3 * dy3);
		
                  //dist2 = 1.0 / (distX * distX + distY * distY);
                                            
		  fx0 += dx0 * d0;
		  fx1 += dx1 * d1;
		  fx2 += dx2 * d2;
                  fx3 += dx3 * d3;
		
                  fy0 += dy0 * d0;
		  fy1 += dy1 * d1;
		  fy2 += dy2 * d2;
                  fy3 += dy3 * d3;
		 
                  //fx += distX * dist2;
                  //fy += distY * dist2;
               }
               
               
               // connected nodes 
               VALUETYPE px0=0, px1=0, px2=0, px3=0, px4=0, px5=0, px6=0, px7=0;
               VALUETYPE py0=0, py1=0, py2=0, py3=0, py4=0, py5=0, py6=0, py7=0;
               
               for(INDEXTYPE j = graph.rowptr[i]; j < graph.rowptr[i+1]; j += 1)
               {
                  int v = graph.colids[j];
                  distX = blasX[v] - x0;
                  distY = blasY[v] - y0;
                  dist = (distX * distX + distY * distY);
                  dist = sqrt(dist) + 1.0 / dist;

                  px0 += distX * dist;
                  py0 += distY * dist;
               }
					
	       for(INDEXTYPE j = graph.rowptr[i+1]; j < graph.rowptr[i+1+1]; j += 1)
               {
                  int v = graph.colids[j];
                  
                  distX = blasX[v] - x1;
                  distY = blasY[v] - y1;
 
                  dist = (distX * distX + distY * distY);
                  dist = sqrt(dist) + 1.0 / dist;

                  px1 += distX * dist;
                  py1 += distY * dist;
               }
				
	       for(INDEXTYPE j = graph.rowptr[i+2]; j < graph.rowptr[i+1+2]; j += 1)
               {
                  int v = graph.colids[j];
                  distX = blasX[v] - x2;
                  distY = blasY[v] - y2;

                  dist = (distX * distX + distY * distY);
                  dist = sqrt(dist) + 1.0 / dist;

                  px2 += distX * dist;
                  py2 += distY * dist;
               }

	       for(INDEXTYPE j = graph.rowptr[i+3]; j < graph.rowptr[i+1+3]; j += 1)
               {
                  int v = graph.colids[j];
                  distX = blasX[v] - x3;
                  distY = blasY[v] - y3;

                  dist = (distX * distX + distY * distY);
                  dist = sqrt(dist) + 1.0 / dist;

                  px3 += distX * dist;
                  py3 += distY * dist;                         
               }

               pb_X[ind] = px0 - fx0;
	       pb_X[ind+1] = px1 - fx1;
	       pb_X[ind+2] = px2 - fx2;
               pb_X[ind+3] = px3 - fx3;
					
	       pb_Y[ind] = py0 - fy0;
	       pb_Y[ind+1] = py1 - fy1;
	       pb_Y[ind+2] = py2 - fy2;
               pb_Y[ind+3] = py3 - fy3;
	    }
	    /*printf("After:\n");
            for(INDEXTYPE i = b * BATCHSIZE; i < (b + 1) * BATCHSIZE; i++)
            {
               printf("%d = %lf,", i, pb_X[i-b*BATCHSIZE]);
            }
	    */
	    for(INDEXTYPE i = b * BATCHSIZE; i < (b + 1) * BATCHSIZE; i += 1)
            {
               int ind = i-b*BATCHSIZE;
               VALUETYPE d;
	       d = (pb_X[ind] * pb_X[ind] + pb_Y[ind] * pb_Y[ind]);
	       ENERGY += d ;
	       
               d = STEP / sqrt(d);
	       blasX[i] += pb_X[ind] * d;
               blasY[i] += pb_Y[ind] * d;
	       
               pb_X[ind] = 0;
	       pb_Y[ind] = 0;    
            }
         }
/*
 *    cleanup 
 */
			INDEXTYPE cleanup = (graph.rows/BATCHSIZE) * BATCHSIZE;
			#pragma omp parallel for schedule(dynamic) 
			for(INDEXTYPE i = cleanup; i < graph.rows; i += 1){
                                INDEXTYPE ind = i- cleanup;
                                VALUETYPE fx = 0, fy = 0, distX, distY, dist, dist2;
                                for(INDEXTYPE j = graph.rowptr[i]; j < graph.rowptr[i+1]; j += 1){
                                        int v = graph.colids[j];
                                        distX = blasX[v] - blasX[i];
                                        distY = blasY[v] - blasY[i];
                                        dist = (distX * distX + distY * distY);
                                        dist = sqrt(dist) + 1.0 / dist;
                                        pb_X[ind] += distX * dist;
                                        pb_Y[ind] += distY * dist;
                                }
                                for(INDEXTYPE j = 0; j < i; j += 1){
                                        distX = blasX[j] - blasX[i];
                                        distY = blasY[j] - blasY[i];
                                        dist2 = 1.0 / (distX * distX + distY * distY);
                                        fx += distX * dist2;
                                        fy += distY * dist2;
                                }
                                for(INDEXTYPE j = i+1; j < graph.rows; j += 1){
                                        distX = blasX[j] - blasX[i];
                                        distY = blasY[j] - blasY[i];
                                        dist2 = 1.0 / (distX * distX + distY * distY);
                                        fx += distX * dist2;
                                        fy += distY * dist2;
                                }
                                pb_X[ind] = pb_X[ind] - fx;
                                pb_Y[ind] = pb_Y[ind] - fy;
                        }	
			for(INDEXTYPE i = cleanup; i < graph.rows; i += 1){
                                int ind = i-cleanup;
                                double dist = (pb_X[ind]*pb_X[ind] + pb_Y[ind]*pb_Y[ind]);
                                ENERGY += dist;
				dist = STEP / sqrt(dist);
				blasX[i] += pb_X[ind] * dist;
                                blasY[i] += pb_Y[ind] * dist;
				pb_X[ind] = pb_Y[ind] = 0;
                        }
                        STEP = STEP * 0.999;
                        LOOP++;
                }
                end = omp_get_wtime();
         #if NO_FWRITE == 0
                cout << "Efficientunroll Minibatch Size:" << BATCHSIZE  << endl;
                cout << "Efficientunroll Minbatch Energy:" << ENERGY << endl;
                cout << "Efficientunroll Minibatch Parallel Wall time required:" << end - start << endl;
               writeToFileEFF("EFFUR"+ to_string(BATCHSIZE)+"PARAOUT" + to_string(LOOP));
         #endif
                result.push_back(ENERGY);
                result.push_back(end - start);
                return result;
        }
   #elif defined (UR2)
/*
 * **************************** UNROLL FACTOR = 2 ************************ 
 */
   vector<VALUETYPE> newalgo::EfficientVersionUnRoll(INDEXTYPE ITERATIONS, 
         INDEXTYPE NUMOFTHREADS, INDEXTYPE BATCHSIZE)
   {
      INDEXTYPE LOOP = 0;
      VALUETYPE start, end, ENERGY, ENERGY0, *pb_X, *pb_Y;
      VALUETYPE STEP = 1.0;
      
      vector<VALUETYPE> result;
      pb_X = static_cast<VALUETYPE *> (::operator new (sizeof(VALUETYPE[BATCHSIZE])));
      pb_Y = static_cast<VALUETYPE *> (::operator new (sizeof(VALUETYPE[BATCHSIZE])));
      ENERGY0 = ENERGY = numeric_limits<VALUETYPE>::max();
     
#if DEGUG == 1
      cout << "*************** Unroll factor = 2" << endl;
      cout << "Rows = " << graph.rows << endl;
#endif

      omp_set_num_threads(NUMOFTHREADS);
/*
 *    time included initDFS 
 */
      start = omp_get_wtime();

      initDFS();
      
      while(LOOP < ITERATIONS)
      {
         ENERGY0 = ENERGY;
         ENERGY = 0;
         for(int i = 0; i < BATCHSIZE; i++)
         {
            pb_X[i] = pb_Y[i] = 0;
         }
	
         
	 for(INDEXTYPE b = 0; b < (graph.rows / BATCHSIZE); b += 1)
         {

/*
 *          NOTE: no cleanup for unroll as lon as BATCHZIE is multiple of unroll
 *          factor 
 */
	    #pragma omp parallel for schedule(static)	
	    for(INDEXTYPE i = b * BATCHSIZE; i < (b + 1) * BATCHSIZE; i += 2)
            {
	       int ind = i-b*BATCHSIZE;

               VALUETYPE distX, distY, dist; 
               VALUETYPE fx0, fx1;
               VALUETYPE fy0, fy1;
               
               
               VALUETYPE x0, x1;
               VALUETYPE y0, y1;
               VALUETYPE d0, d1;
					
	       x0 = blasX[i];
	       x1 = blasX[i+1];

	       y0 = blasY[i];
	       y1 = blasY[i+1];	
					
	       fx0 = fx1 = 0;
	       fy0 = fy1 = 0;		
		
/*
 *              j is up to i... lower up case  
 */
               for(INDEXTYPE j = 0; j < i; j += 1)
               {
                  VALUETYPE dx0, dx1;
                  VALUETYPE dy0, dy1;
                  VALUETYPE xj = blasX[j];
                  VALUETYPE yj = blasY[j];
		  
                  dx0 = xj - x0;
		  dx1 = xj - x1;
	          
                  dy0 = yj - y0;
		  dy1 = yj - y1;
		
                  //distX = blasX[j] - blasX[i];
                  //distY = blasY[j] - blasY[i];
                                   	
	          d0 = 1.0 / (dx0 * dx0 + dy0 * dy0);
		  d1 = 1.0 / (dx1 * dx1 + dy1 * dy1);
		
                  //dist2 = 1.0 / (distX * distX + distY * distY);
                                            
		  fx0 += dx0 * d0;
		  fx1 += dx1 * d1;
		
                  fy0 += dy0 * d0;
		  fy1 += dy1 * d1;
		 
                  //fx += distX * dist2;
                  //fy += distY * dist2;
               }		
/*
 *             location where we need to skip some points i==j
 *             NOTE: its UR iteration, no need to optimize 
 *             Hopefully compiler will unroll and get rid of the conditions
 */
               for (INDEXTYPE j = i; j < i+2; j++) 
               {
                  VALUETYPE xj = blasX[j];
                  VALUETYPE yj = blasY[j];
		  
                  if (j != i)
                  {
                     VALUETYPE dx0, dy0;
                     dx0 = xj - x0;
                     dy0 = yj - y0;
	             d0 = 1.0 / (dx0 * dx0 + dy0 * dy0);
		     fx0 += dx0 * d0;
                     fy0 += dy0 * d0;
                  }
                  if ( j != i+1 )
                  {
                     VALUETYPE dx1, dy1;
                     dx1 = xj - x1;
                     dy1 = yj - y1;
	             d1 = 1.0 / (dx1 * dx1 + dy1 * dy1);
		     fx1 += dx1 * d1;
                     fy1 += dy1 * d1;
                  }
               }
/*
 *             Upper part  
 *
 */ 
               for(INDEXTYPE j = i+2; j < graph.rows; j += 1)
               {
                  VALUETYPE dx0, dx1;
                  VALUETYPE dy0, dy1;
                  VALUETYPE xj = blasX[j];
                  VALUETYPE yj = blasY[j];
		  
                  dx0 = xj - x0;
		  dx1 = xj - x1;
	          
                  dy0 = yj - y0;
		  dy1 = yj - y1;
		
                  //distX = blasX[j] - blasX[i];
                  //distY = blasY[j] - blasY[i];
                                   	
	          d0 = 1.0 / (dx0 * dx0 + dy0 * dy0);
		  d1 = 1.0 / (dx1 * dx1 + dy1 * dy1);
		
                  //dist2 = 1.0 / (distX * distX + distY * distY);
                                            
		  fx0 += dx0 * d0;
		  fx1 += dx1 * d1;
		
                  fy0 += dy0 * d0;
		  fy1 += dy1 * d1;
		 
                  //fx += distX * dist2;
                  //fy += distY * dist2;
               }
               
               
               // connected nodes 
               VALUETYPE px0=0, px1=0, px2=0, px3=0, px4=0, px5=0, px6=0, px7=0;
               VALUETYPE py0=0, py1=0, py2=0, py3=0, py4=0, py5=0, py6=0, py7=0;
               
               for(INDEXTYPE j = graph.rowptr[i]; j < graph.rowptr[i+1]; j += 1)
               {
                  int v = graph.colids[j];
                  distX = blasX[v] - x0;
                  distY = blasY[v] - y0;
                  dist = (distX * distX + distY * distY);
                  dist = sqrt(dist) + 1.0 / dist;

                  px0 += distX * dist;
                  py0 += distY * dist;
               }
					
	       for(INDEXTYPE j = graph.rowptr[i+1]; j < graph.rowptr[i+1+1]; j += 1)
               {
                  int v = graph.colids[j];
                  
                  distX = blasX[v] - x1;
                  distY = blasY[v] - y1;
 
                  dist = (distX * distX + distY * distY);
                  dist = sqrt(dist) + 1.0 / dist;

                  px1 += distX * dist;
                  py1 += distY * dist;
               }
				
               pb_X[ind] = px0 - fx0;
	       pb_X[ind+1] = px1 - fx1;
					
	       pb_Y[ind] = py0 - fy0;
	       pb_Y[ind+1] = py1 - fy1;
	    }
	    /*printf("After:\n");
            for(INDEXTYPE i = b * BATCHSIZE; i < (b + 1) * BATCHSIZE; i++)
            {
               printf("%d = %lf,", i, pb_X[i-b*BATCHSIZE]);
            }
	    */
	    for(INDEXTYPE i = b * BATCHSIZE; i < (b + 1) * BATCHSIZE; i += 1)
            {
               int ind = i-b*BATCHSIZE;
               VALUETYPE d;
	       d = (pb_X[ind] * pb_X[ind] + pb_Y[ind] * pb_Y[ind]);
	       ENERGY += d ;
	       
               d = STEP / sqrt(d);
	       blasX[i] += pb_X[ind] * d;
               blasY[i] += pb_Y[ind] * d;
	       
               pb_X[ind] = 0;
	       pb_Y[ind] = 0;    
            }
         }
/*
 *    cleanup 
 */
			INDEXTYPE cleanup = (graph.rows/BATCHSIZE) * BATCHSIZE;
            #if DEBUG == 1
                        cout << "Total cleanup nodes = " << graph.rows - cleanup;
            #endif
			#pragma omp parallel for schedule(dynamic) 
			for(INDEXTYPE i = cleanup; i < graph.rows; i += 1){
                                INDEXTYPE ind = i- cleanup;
                                VALUETYPE fx = 0, fy = 0, distX, distY, dist, dist2;
                                for(INDEXTYPE j = graph.rowptr[i]; j < graph.rowptr[i+1]; j += 1){
                                        int v = graph.colids[j];
                                        distX = blasX[v] - blasX[i];
                                        distY = blasY[v] - blasY[i];
                                        dist = (distX * distX + distY * distY);
                                        dist = sqrt(dist) + 1.0 / dist;
                                        pb_X[ind] += distX * dist;
                                        pb_Y[ind] += distY * dist;
                                }
                                for(INDEXTYPE j = 0; j < i; j += 1){
                                        distX = blasX[j] - blasX[i];
                                        distY = blasY[j] - blasY[i];
                                        dist2 = 1.0 / (distX * distX + distY * distY);
                                        fx += distX * dist2;
                                        fy += distY * dist2;
                                }
                                for(INDEXTYPE j = i+1; j < graph.rows; j += 1){
                                        distX = blasX[j] - blasX[i];
                                        distY = blasY[j] - blasY[i];
                                        dist2 = 1.0 / (distX * distX + distY * distY);
                                        fx += distX * dist2;
                                        fy += distY * dist2;
                                }
                                pb_X[ind] = pb_X[ind] - fx;
                                pb_Y[ind] = pb_Y[ind] - fy;
                        }	
			for(INDEXTYPE i = cleanup; i < graph.rows; i += 1){
                                int ind = i-cleanup;
                                double dist = (pb_X[ind]*pb_X[ind] + pb_Y[ind]*pb_Y[ind]);
                                ENERGY += dist;
				dist = STEP / sqrt(dist);
				blasX[i] += pb_X[ind] * dist;
                                blasY[i] += pb_Y[ind] * dist;
				pb_X[ind] = pb_Y[ind] = 0;
                        }
                        STEP = STEP * 0.999;
                        LOOP++;
                }
                end = omp_get_wtime();
         #if NO_FWRITE == 0
                cout << "Efficientunroll Minibatch Size:" << BATCHSIZE  << endl;
                cout << "Efficientunroll Minbatch Energy:" << ENERGY << endl;
                cout << "Efficientunroll Minibatch Parallel Wall time required:" << end - start << endl;
                writeToFileEFF("EFFUR"+ to_string(BATCHSIZE)+"PARAOUT" + to_string(LOOP));
         #endif
                result.push_back(ENERGY);
                result.push_back(end - start);
                return result;
        }
   #elif defined(UR1)
/*
 * **************************** UNROLL FACTOR = 1 ************************ 
 */
   vector<VALUETYPE> newalgo::EfficientVersionUnRoll(INDEXTYPE ITERATIONS, 
         INDEXTYPE NUMOFTHREADS, INDEXTYPE BATCHSIZE)
   {
      INDEXTYPE LOOP = 0;
      VALUETYPE start, end, ENERGY, ENERGY0, *pb_X, *pb_Y;
      VALUETYPE STEP = 1.0;
      
      vector<VALUETYPE> result;
      pb_X = static_cast<VALUETYPE *> (::operator new (sizeof(VALUETYPE[BATCHSIZE])));
      pb_Y = static_cast<VALUETYPE *> (::operator new (sizeof(VALUETYPE[BATCHSIZE])));
      ENERGY0 = ENERGY = numeric_limits<VALUETYPE>::max();
     
#if 0
      cout << "Rows = " << graph.rows << endl;
      exit(1);
#endif

      omp_set_num_threads(NUMOFTHREADS);
/*
 *    time included initDFS 
 */
      start = omp_get_wtime();

      initDFS();
      
      while(LOOP < ITERATIONS)
      {
         ENERGY0 = ENERGY;
         ENERGY = 0;
         for(int i = 0; i < BATCHSIZE; i++)
         {
            pb_X[i] = pb_Y[i] = 0;
         }
	
         
	 for(INDEXTYPE b = 0; b < (graph.rows / BATCHSIZE); b += 1)
         {

/*
 *          NOTE: no cleanup for unroll as lon as BATCHZIE is multiple of unroll
 *          factor 
 */
	    #pragma omp parallel for schedule(static)	
	    for(INDEXTYPE i = b * BATCHSIZE; i < (b + 1) * BATCHSIZE; i += 1)
            {
	       int ind = i-b*BATCHSIZE;

               VALUETYPE distX, distY, dist; 
               VALUETYPE fx0;
               VALUETYPE fy0;
               
               
               VALUETYPE x0;
               VALUETYPE y0;
               VALUETYPE d0;
					
	       x0 = blasX[i];

	       y0 = blasY[i];
					
	       fx0 = 0;
	       fy0 = 0;		
		
/*
 *              j is up to i... lower up case  
 */
               for(INDEXTYPE j = 0; j < i; j += 1)
               {
                  VALUETYPE dx0, dx1;
                  VALUETYPE dy0, dy1;
                  VALUETYPE xj = blasX[j];
                  VALUETYPE yj = blasY[j];
		  
                  dx0 = xj - x0;
                  dy0 = yj - y0;
		
                  //distX = blasX[j] - blasX[i];
                  //distY = blasY[j] - blasY[i];
                                   	
	          d0 = 1.0 / (dx0 * dx0 + dy0 * dy0);
		
                  //dist2 = 1.0 / (distX * distX + distY * distY);
                                            
		  fx0 += dx0 * d0;
                  fy0 += dy0 * d0;
		 
                  //fx += distX * dist2;
                  //fy += distY * dist2;
               }		
/*
 *             location where we need to skip some points i==j
 *             NOTE: its UR iteration, no need to optimize 
 *             Hopefully compiler will unroll and get rid of the conditions
 */
/*
 *             Upper part  
 *
 */ 
               for(INDEXTYPE j = i+1; j < graph.rows; j += 1)
               {
                  VALUETYPE dx0, dx1;
                  VALUETYPE dy0, dy1;
                  VALUETYPE xj = blasX[j];
                  VALUETYPE yj = blasY[j];
		  
                  dx0 = xj - x0;
                  dy0 = yj - y0;
		
                  //distX = blasX[j] - blasX[i];
                  //distY = blasY[j] - blasY[i];
                                   	
	          d0 = 1.0 / (dx0 * dx0 + dy0 * dy0);
		
                  //dist2 = 1.0 / (distX * distX + distY * distY);
                                            
		  fx0 += dx0 * d0;
		
                  fy0 += dy0 * d0;
		 
                  //fx += distX * dist2;
                  //fy += distY * dist2;
               }
               
               
               // connected nodes 
               VALUETYPE px0=0, px1=0, px2=0, px3=0, px4=0, px5=0, px6=0, px7=0;
               VALUETYPE py0=0, py1=0, py2=0, py3=0, py4=0, py5=0, py6=0, py7=0;
               
               for(INDEXTYPE j = graph.rowptr[i]; j < graph.rowptr[i+1]; j += 1)
               {
                  int v = graph.colids[j];
                  distX = blasX[v] - x0;
                  distY = blasY[v] - y0;
                  dist = (distX * distX + distY * distY);
                  dist = sqrt(dist) + 1.0 / dist;

                  px0 += distX * dist;
                  py0 += distY * dist;
               }
				
               pb_X[ind] = px0 - fx0;
	       pb_Y[ind] = py0 - fy0;
	    }
	    /*printf("After:\n");
            for(INDEXTYPE i = b * BATCHSIZE; i < (b + 1) * BATCHSIZE; i++)
            {
               printf("%d = %lf,", i, pb_X[i-b*BATCHSIZE]);
            }
	    */
	    for(INDEXTYPE i = b * BATCHSIZE; i < (b + 1) * BATCHSIZE; i += 1)
            {
               int ind = i-b*BATCHSIZE;
               VALUETYPE d;
	       d = (pb_X[ind] * pb_X[ind] + pb_Y[ind] * pb_Y[ind]);
	       ENERGY += d ;
	       
               d = STEP / sqrt(d);
	       blasX[i] += pb_X[ind] * d;
               blasY[i] += pb_Y[ind] * d;
	       
               pb_X[ind] = 0;
	       pb_Y[ind] = 0;    
            }
         }
/*
 *    cleanup 
 */
			INDEXTYPE cleanup = (graph.rows/BATCHSIZE) * BATCHSIZE;
			#pragma omp parallel for schedule(dynamic) 
			for(INDEXTYPE i = cleanup; i < graph.rows; i += 1){
                                INDEXTYPE ind = i- cleanup;
                                VALUETYPE fx = 0, fy = 0, distX, distY, dist, dist2;
                                for(INDEXTYPE j = graph.rowptr[i]; j < graph.rowptr[i+1]; j += 1){
                                        int v = graph.colids[j];
                                        distX = blasX[v] - blasX[i];
                                        distY = blasY[v] - blasY[i];
                                        dist = (distX * distX + distY * distY);
                                        dist = sqrt(dist) + 1.0 / dist;
                                        pb_X[ind] += distX * dist;
                                        pb_Y[ind] += distY * dist;
                                }
                                for(INDEXTYPE j = 0; j < i; j += 1){
                                        distX = blasX[j] - blasX[i];
                                        distY = blasY[j] - blasY[i];
                                        dist2 = 1.0 / (distX * distX + distY * distY);
                                        fx += distX * dist2;
                                        fy += distY * dist2;
                                }
                                for(INDEXTYPE j = i+1; j < graph.rows; j += 1){
                                        distX = blasX[j] - blasX[i];
                                        distY = blasY[j] - blasY[i];
                                        dist2 = 1.0 / (distX * distX + distY * distY);
                                        fx += distX * dist2;
                                        fy += distY * dist2;
                                }
                                pb_X[ind] = pb_X[ind] - fx;
                                pb_Y[ind] = pb_Y[ind] - fy;
                        }	
			for(INDEXTYPE i = cleanup; i < graph.rows; i += 1){
                                int ind = i-cleanup;
                                double dist = (pb_X[ind]*pb_X[ind] + pb_Y[ind]*pb_Y[ind]);
                                ENERGY += dist;
				dist = STEP / sqrt(dist);
				blasX[i] += pb_X[ind] * dist;
                                blasY[i] += pb_Y[ind] * dist;
				pb_X[ind] = pb_Y[ind] = 0;
                        }
                        STEP = STEP * 0.999;
                        LOOP++;
                }
                end = omp_get_wtime();
         #if NO_FWRITE == 0
                cout << "Efficientunroll Minibatch Size:" << BATCHSIZE  << endl;
                cout << "Efficientunroll Minbatch Energy:" << ENERGY << endl;
                cout << "Efficientunroll Minibatch Parallel Wall time required:" << end - start << endl;
                writeToFileEFF("EFFUR"+ to_string(BATCHSIZE)+"PARAOUT" + to_string(LOOP));
         #endif
                result.push_back(ENERGY);
                result.push_back(end - start);
                return result;
        }
   #elif defined(UR4_MEMOPT)
/*
 * **************************** MEMOPT: UNROLL FACTOR = 4 **************** 
 */
   vector<VALUETYPE> newalgo::EfficientVersionUnRoll(INDEXTYPE ITERATIONS, 
         INDEXTYPE NUMOFTHREADS, INDEXTYPE BATCHSIZE)
   {
      INDEXTYPE LOOP = 0;
      VALUETYPE start, end, ENERGY, ENERGY0, *pb_X, *pb_Y;
      VALUETYPE STEP = 1.0;
      
      vector<VALUETYPE> result;
      pb_X = static_cast<VALUETYPE *> (::operator new (sizeof(VALUETYPE[BATCHSIZE])));
      pb_Y = static_cast<VALUETYPE *> (::operator new (sizeof(VALUETYPE[BATCHSIZE])));
      ENERGY0 = ENERGY = numeric_limits<VALUETYPE>::max();
     
#if 0
      cout << "Rows = " << graph.rows << endl;
      exit(1);
#endif

      omp_set_num_threads(NUMOFTHREADS);
/*
 *    time included initDFS 
 */
      start = omp_get_wtime();

      initDFS();
      
      while(LOOP < ITERATIONS)
      {
         ENERGY0 = ENERGY;
         ENERGY = 0;
         for(int i = 0; i < BATCHSIZE; i++)
         {
            pb_X[i] = pb_Y[i] = 0;
         }
	
         
	 for(INDEXTYPE b = 0; b < (graph.rows / BATCHSIZE); b += 1)
         {

/*
 *          NOTE: no cleanup for unroll as lon as BATCHZIE is multiple of unroll
 *          factor
 *          NOTE: calling omp parallel for has overhead for fork joining 
 *          multiple times... so, handle loop manually 
 *
 *          implicit unrolling = 4 * 8 = 32 .. 
 *          per thread slice = (256 / 32) / 18 = .... doesn't make sense???
 */
            #pragma omp parallel 
            {
               int id, nthreads, chunksize;

               id = omp_get_thread_num();
               nthreads = omp_get_num_threads();
               chunksize = BATCHSIZE / nthreads; 

	       for(INDEXTYPE i = b * BATCHSIZE; i < (b + 1) * BATCHSIZE; i += 4)
               {
	          int ind = i-b*BATCHSIZE;

                  VALUETYPE distX, distY, dist; 
                  VALUETYPE fx0, fx1, fx2, fx3;
                  VALUETYPE fy0, fy1, fy2, fy3;
                  VALUETYPE x0, x1, x2, x3;
                  VALUETYPE y0, y1, y2, y3;
                  VALUETYPE d0, d1, d2, d3;
					
	          x0 = blasX[i];
	          x1 = blasX[i+1];
	          x2 = blasX[i+2];
	          x3 = blasX[i+3];

	          y0 = blasY[i];
	          y1 = blasY[i+1];	
	          y2 = blasY[i+2];
	          y3 = blasY[i+3];
					
	          fx0 = fx1 = fx2 = fx3 = 0;
	          fy0 = fy1 = fy2 = fy3 = 0;		
/*
 *              j is up to i... lower up case  
 */
                  for(INDEXTYPE j = 0; j < i; j += 1)
                  {
                     VALUETYPE dx0, dx1, dx2, dx3;
                     VALUETYPE dy0, dy1, dy2, dy3;
                     VALUETYPE xj = blasX[j];
                     VALUETYPE yj = blasY[j];
		  
                     dx0 = xj - x0;
		     dx1 = xj - x1;
		     dx2 = xj - x2;
                     dx3 = xj - x3;
	          
                     dy0 = yj - y0;
		     dy1 = yj - y1;
		     dy2 = yj - y2;
                     dy3 = yj - y3;
                                   	
	             d0 = 1.0 / (dx0 * dx0 + dy0 * dy0);
		     d1 = 1.0 / (dx1 * dx1 + dy1 * dy1);
		     d2 = 1.0 / (dx2 * dx2 + dy2 * dy2);
		     d3 = 1.0 / (dx3 * dx3 + dy3 * dy3);
		
		     fx0 += dx0 * d0;
		     fx1 += dx1 * d1;
		     fx2 += dx2 * d2;
                     fx3 += dx3 * d3;
		
                     fy0 += dy0 * d0;
		     fy1 += dy1 * d1;
		     fy2 += dy2 * d2;
                     fy3 += dy3 * d3;
                  }		
/*
 *                location where we need to skip some points i==j
 *                NOTE: its UR iteration, no need to optimize 
 *                Hopefully compiler will unroll and get rid of the conditions
 *                NOTE: can be optimized it using K-reg with intrinsic
 */
                  for (INDEXTYPE j = i; j < i+4; j++) 
                  {
                     VALUETYPE xj = blasX[j];
                     VALUETYPE yj = blasY[j];
		  
                     if (j != i)
                     {
                        VALUETYPE dx0, dy0;
                        dx0 = xj - x0;
                        dy0 = yj - y0;
	                d0 = 1.0 / (dx0 * dx0 + dy0 * dy0);
		        fx0 += dx0 * d0;
                        fy0 += dy0 * d0;
                     }
                     if ( j != i+1 )
                     {
                        VALUETYPE dx1, dy1;
                        dx1 = xj - x1;
                        dy1 = yj - y1;
	                d1 = 1.0 / (dx1 * dx1 + dy1 * dy1);
		        fx1 += dx1 * d1;
                        fy1 += dy1 * d1;
                     }
                     if ( j != i+2 )
                     {
                        VALUETYPE dx2, dy2;
                        dx2 = xj - x2;
                        dy2 = yj - y2;
	                d2 = 1.0 / (dx2 * dx2 + dy2 * dy2);
		        fx2 += dx2 * d2;
                        fy2 += dy2 * d2;
                     }
                     if ( j != i+3 )
                     {
                        VALUETYPE dx3, dy3;
                        dx3 = xj - x3;
                        dy3 = yj - y3;
	                d3 = 1.0 / (dx3 * dx3 + dy3 * dy3);
		        fx3 += dx3 * d3;
                        fy3 += dy3 * d3;
                     }
                  }
/*
 *                Upper part  
 */ 
                  for(INDEXTYPE j = i+4; j < graph.rows; j += 1)
                  {
                     VALUETYPE dx0, dx1, dx2, dx3;
                     VALUETYPE dy0, dy1, dy2, dy3;
                     VALUETYPE xj = blasX[j];
                     VALUETYPE yj = blasY[j];
		  
                     dx0 = xj - x0;
		     dx1 = xj - x1;
		     dx2 = xj - x2;
                     dx3 = xj - x3;
	          
                     dy0 = yj - y0;
		     dy1 = yj - y1;
		     dy2 = yj - y2;
                     dy3 = yj - y3;
                                   	
	             d0 = 1.0 / (dx0 * dx0 + dy0 * dy0);
		     d1 = 1.0 / (dx1 * dx1 + dy1 * dy1);
		     d2 = 1.0 / (dx2 * dx2 + dy2 * dy2);
		     d3 = 1.0 / (dx3 * dx3 + dy3 * dy3);
		
		     fx0 += dx0 * d0;
		     fx1 += dx1 * d1;
		     fx2 += dx2 * d2;
                     fx3 += dx3 * d3;
		
                     fy0 += dy0 * d0;
		     fy1 += dy1 * d1;
		     fy2 += dy2 * d2;
                     fy3 += dy3 * d3;
                  }
                  pb_X[ind] = fx0;
	          pb_X[ind+1] = fx1;
	          pb_X[ind+2] = fx2;
                  pb_X[ind+3] = fx3;
					
	          pb_Y[ind] = fy0;
	          pb_Y[ind+1] = fy1;
	          pb_Y[ind+2] = fy2;
                  pb_Y[ind+3] = fy3;
	       }
/*
 *          Remove connected node calc out of the nested loop
 *          FIXME: ***** the code slows down by 10% 
 *
 */
#if 0
	    #pragma omp parallel for schedule(static)	
	    for(INDEXTYPE i = b * BATCHSIZE; i < (b + 1) * BATCHSIZE; i += 1)
            {
               int ind = i-b*BATCHSIZE;
               VALUETYPE pbX=0.0, pbY=0.0;
               VALUETYPE dist, distX, distY;
               for(INDEXTYPE j = graph.rowptr[i]; j < graph.rowptr[i+1]; j += 1)
               {
                  int v = graph.colids[j];
                  distX = blasX[v] - blasX[i];
                  distY = blasY[v] - blasY[i];
                  dist = (distX * distX + distY * distY);
                  dist = sqrt(dist) + 1.0 / dist;
                  pbX += distX * dist;
                  pbY += distY * dist;
               }

               pbX = pbX - pb_X[ind];
               pbY = pbY - pb_Y[ind];
	       
               dist = (pbX * pbX + pbY * pbY);
               #pragma omp atomic
               ENERGY += dist ;
	       
               dist = STEP / sqrt(dist);
	       blasX[i] += pbX * dist;
               blasY[i] += pbY * dist;
               pb_X[ind] = 0;
	       pb_Y[ind] = 0;    
            }
#else
/*
 *          Calc connected nodes 
 *          complexity : BATCHSIZE * avg connection 
 *          ************ still 10% slow, why ????? 
 *          Unrolling by 4, we eliminate 4 * rows memory loads, 
 *          but moving out the connected nodes, we have extra loads 
 *          = BATCHSIZE * avg connection
 *          ***** Need to find:  4*VLEN*rows >> BATCHSIZE * avg connection
 *                         example: pkustk01
 *                                  avg connect = 50 (guessing from prof)
 *                               4*8*22044  >> 256 * 50
 *                               705408     >> 12800 
 *
 */
	       //#pragma omp parallel for schedule(static)	
	       for(INDEXTYPE i = b * BATCHSIZE; i < (b + 1) * BATCHSIZE; i += 1)
               {
                  int ind = i-b*BATCHSIZE;
                  VALUETYPE pbX=0.0, pbY=0.0;
                  VALUETYPE dist, distX, distY;
                  for(INDEXTYPE j = graph.rowptr[i]; j < graph.rowptr[i+1]; j += 1)
                  {
                     int v = graph.colids[j];
                     distX = blasX[v] - blasX[i];
                     distY = blasY[v] - blasY[i];
                     dist = (distX * distX + distY * distY);
                     dist = sqrt(dist) + 1.0 / dist;
                     pbX += distX * dist;
                     pbY += distY * dist;
                  }
                  pb_X[ind] = pbX - pb_X[ind];
                  pb_Y[ind] = pbY - pb_Y[ind];
               }
            }
/*
 *          Update nodes 
 */
	    for(INDEXTYPE i = b * BATCHSIZE; i < (b + 1) * BATCHSIZE; i += 1)
            {
               int ind = i-b*BATCHSIZE;
               VALUETYPE d;
	       d = (pb_X[ind] * pb_X[ind] + pb_Y[ind] * pb_Y[ind]);
	       ENERGY += d ;
	       
               d = STEP / sqrt(d);
	       blasX[i] += pb_X[ind] * d;
               blasY[i] += pb_Y[ind] * d;
	       
               pb_X[ind] = 0;
	       pb_Y[ind] = 0;    
            }
#endif
         }
/*
 *    cleanup 
 */
			INDEXTYPE cleanup = (graph.rows/BATCHSIZE) * BATCHSIZE;
			#pragma omp parallel for schedule(dynamic) 
			for(INDEXTYPE i = cleanup; i < graph.rows; i += 1){
                                INDEXTYPE ind = i- cleanup;
                                VALUETYPE fx = 0, fy = 0, distX, distY, dist, dist2;
                                for(INDEXTYPE j = graph.rowptr[i]; j < graph.rowptr[i+1]; j += 1){
                                        int v = graph.colids[j];
                                        distX = blasX[v] - blasX[i];
                                        distY = blasY[v] - blasY[i];
                                        dist = (distX * distX + distY * distY);
                                        dist = sqrt(dist) + 1.0 / dist;
                                        pb_X[ind] += distX * dist;
                                        pb_Y[ind] += distY * dist;
                                }
                                for(INDEXTYPE j = 0; j < i; j += 1){
                                        distX = blasX[j] - blasX[i];
                                        distY = blasY[j] - blasY[i];
                                        dist2 = 1.0 / (distX * distX + distY * distY);
                                        fx += distX * dist2;
                                        fy += distY * dist2;
                                }
                                for(INDEXTYPE j = i+1; j < graph.rows; j += 1){
                                        distX = blasX[j] - blasX[i];
                                        distY = blasY[j] - blasY[i];
                                        dist2 = 1.0 / (distX * distX + distY * distY);
                                        fx += distX * dist2;
                                        fy += distY * dist2;
                                }
                                pb_X[ind] = pb_X[ind] - fx;
                                pb_Y[ind] = pb_Y[ind] - fy;
                        }	
			for(INDEXTYPE i = cleanup; i < graph.rows; i += 1){
                                int ind = i-cleanup;
                                double dist = (pb_X[ind]*pb_X[ind] + pb_Y[ind]*pb_Y[ind]);
                                ENERGY += dist;
				dist = STEP / sqrt(dist);
				blasX[i] += pb_X[ind] * dist;
                                blasY[i] += pb_Y[ind] * dist;
				pb_X[ind] = pb_Y[ind] = 0;
                        }
                        STEP = STEP * 0.999;
                        LOOP++;
                }
                end = omp_get_wtime();
         
         #if NO_FWRITE == 0
                cout << "Efficientunroll Minibatch Size:" << BATCHSIZE  << endl;
                cout << "Efficientunroll Minbatch Energy:" << ENERGY << endl;
                cout << "Efficientunroll Minibatch Parallel Wall time required:" << end - start << endl;
               writeToFileEFF("EFFUR"+ to_string(BATCHSIZE)+"PARAOUT" + to_string(LOOP));
         #endif
                result.push_back(ENERGY);
                result.push_back(end - start);
                return result;
        }

   #endif

#endif

#if 0

	vector<VALUETYPE> newalgo::EfficientVersionV2(INDEXTYPE ITERATIONS, INDEXTYPE NUMOFTHREADS, INDEXTYPE BATCHSIZE){
                INDEXTYPE LOOP = 0;
                VALUETYPE start, end, ENERGY, ENERGY0, *pb_X, *pb_Y;
                VALUETYPE STEP = 1.0;
                vector<VALUETYPE> result;
                pb_X = static_cast<VALUETYPE *> (::operator new (sizeof(VALUETYPE[BATCHSIZE])));
                pb_Y = static_cast<VALUETYPE *> (::operator new (sizeof(VALUETYPE[BATCHSIZE])));
                ENERGY0 = ENERGY = numeric_limits<VALUETYPE>::max();
                omp_set_num_threads(NUMOFTHREADS);
                start = omp_get_wtime();
                initDFS();
                while(LOOP < ITERATIONS){
                        ENERGY0 = ENERGY;
                        ENERGY = 0;
                        for(int i = 0; i < BATCHSIZE; i++){
                                pb_X[i] = pb_Y[i] = 0;
                        }
			for(INDEXTYPE b = 0; b < (graph.rows / BATCHSIZE); b += 1){
                                //#pragma omp parallel for schedule(static)
                                for(INDEXTYPE i = b * BATCHSIZE; i < (b + 1) * BATCHSIZE; i += 8){
                                        VALUETYPE x, y, fx = 0, fy = 0, distX, distY, dist, dist2;
					register VTYPE vbx0, vbx1, vbx2, vbx3, vbx4, vbx5, vbx6, vbx7, dc;
					register VTYPE vby0, vby1, vby2, vby3, vby4, vby5, vby6, vby7;
					register VTYPE vfx0, vfx1, vfx2, vfx3, vfx4, vfx5, vfx6, vfx7;
                                        register VTYPE vfy0, vfy1, vfy2, vfy3, vfy4, vfy5, vfy6, vfy7;
					
					vfx0 = _mm512_set1_pd(0.0);
					vfy0 = _mm512_set1_pd(0.0);
					dc = _mm512_set1_pd(1.0);

					vbx0 = _mm512_loadu_pd(blasX + i);
					vby0 = _mm512_loadu_pd(blasY + i);
                                        
					int ind = i-b*BATCHSIZE;
                                        for(INDEXTYPE j = 0; j < i; j += 1){
                                                register VTYPE vcx, vcy, d0, d1, d2, d3, d4, d5, d6, d7;
							
						vcx = _mm512_set1_pd(blasX[j]);
						vcy = _mm512_set1_pd(blasY[j]);
						
						vbx0 = _mm512_sub_pd(vcx, vbx0);
						vby0 = _mm512_sub_pd(vcy, vby0);
						//distX = blasX[j] - blasX[i];
                                                //distY = blasY[j] - blasY[i];
                                                
						d0 = _mm512_mul_pd(vbx0, vbx0);
						d0 = _mm512_fmadd_pd(vby0, vby0, d0);
						d0 = _mm512_div_pd(dc, d0);
						//dist2 = 1.0 / (distX * distX + distY * distY);
                                                
						vfx0 = _mm512_fmadd_pd(vbx0, d0, vfx0);
						vfy0 = _mm512_fmadd_pd(vby0, d0, vfy0);
						//fx += distX * dist2;
                                                //fy += distY * dist2;
                                        }
                                        for(INDEXTYPE j = i+1; j < graph.rows; j += 1){
                                                register VTYPE vcx, vcy, d0, d1, d2, d3, d4, d5, d6, d7;

                                                vcx = _mm512_set1_pd(blasX[j]);
                                                vcy = _mm512_set1_pd(blasY[j]);

                                                vbx0 = _mm512_sub_pd(vcx, vbx0);
                                                vby0 = _mm512_sub_pd(vcy, vby0);
						//distX = blasX[j] - blasX[i];
                                                //distY = blasY[j] - blasY[i];
                                                
						d0 = _mm512_mul_pd(vbx0, vbx0);
                                                d0 = _mm512_fmadd_pd(vby0, vby0, d0);
                                                d0 = _mm512_div_pd(dc, d0);
						//dist2 = 1.0 / (distX * distX + distY * distY);
                                                
						vfx0 = _mm512_fmadd_pd(vbx0, d0, vfx0);
                                                vfy0 = _mm512_fmadd_pd(vby0, d0, vfy0);
						//fx += distX * dist2;
                                                //fy += distY * dist2;
                                        }
					_mm512_storeu_pd(pb_X+ind, vfx0);
					_mm512_storeu_pd(pb_Y+ind, vfy0);
					for(INDEXTYPE j = graph.rowptr[i]; j < graph.rowptr[i+1]; j += 1){
                                                int v = graph.colids[j];
                                                distX = blasX[v] - blasX[i];
                                                distY = blasY[v] - blasY[i];
                                              	dist2 = (distX * distX + distY * distY);
						dist = sqrt(dist2) + 1.0 / (dist2);
						pb_X[ind] += distX * dist;
                                                pb_Y[ind] += distY * dist;
                                        }
                                        //pb_X[ind] = pb_X[ind] - fx;
                                        //pb_Y[ind] = pb_Y[ind] - fy;
                                }
				printf("After:\n");
				for(INDEXTYPE i = b * BATCHSIZE; i < (b + 1) * BATCHSIZE; i++){
					printf("%d = %lf,", i, pb_X[i-b*BATCHSIZE]);
				}
				printf("\n");
				for(INDEXTYPE i = b * BATCHSIZE; i < (b + 1) * BATCHSIZE; i++){
                                        int ind = i-b*BATCHSIZE;
					double dist2 = pb_X[ind]*pb_X[ind] + pb_Y[ind]*pb_Y[ind];
                                        double dist = 1.0 / sqrt(dist2);
                                        blasX[i] += pb_X[ind] * STEP * dist;
                                        blasY[i] += pb_Y[ind] * STEP * dist;
                                        ENERGY += (dist2);
                                }
                        }
			INDEXTYPE cleanup = (graph.rows/BATCHSIZE) * BATCHSIZE;
                        //#pragma omp parallel for schedule(static)
                        for(INDEXTYPE i = cleanup; i < graph.rows; i += 1){
                                INDEXTYPE ind = i- cleanup;
                                VALUETYPE fx = 0, fy = 0, distX, distY, dist, dist2;
                                for(INDEXTYPE j = graph.rowptr[i]; j < graph.rowptr[i+1]; j += 1){
                                        int v = graph.colids[j];
                                        distX = blasX[v] - blasX[i];
                                        distY = blasY[v] - blasY[i];
                                        dist2 = distX * distX + distY * distY;
                                        dist = sqrt(dist2) + 1.0 / dist2;
                                        pb_X[ind] += distX * dist;
                                        pb_Y[ind] += distY * dist;
                                }
                                for(INDEXTYPE j = 0; j < i; j += 1){
                                        distX = blasX[j] - blasX[i];
                                        distY = blasY[j] - blasY[i];
                                        dist2 = 1.0 / (distX * distX + distY * distY);
                                        fx += distX * dist2;
                                        fy += distY * dist2;
                                }
                                for(INDEXTYPE j = i+1; j < graph.rows; j += 1){
                                        distX = blasX[j] - blasX[i];
                                        distY = blasY[j] - blasY[i];
                                        dist2 = 1.0 / (distX * distX + distY * distY);
                                        fx += distX * dist2;
                                        fy += distY * dist2;
                                }
                                pb_X[ind] = pb_X[ind] - fx;
                                pb_Y[ind] = pb_Y[ind] - fy;
                        }
			for(INDEXTYPE i = cleanup; i < graph.rows; i += 1){
                                int ind = i-cleanup;
                                double dist = (1.0 * STEP) / sqrt(pb_X[ind]*pb_X[ind] + pb_Y[ind]*pb_Y[ind]);
                                blasX[i] += pb_X[ind] * dist;
                                blasY[i] += pb_Y[ind] * dist;
                                ENERGY += (pb_X[ind] * pb_X[ind] + pb_Y[ind] * pb_Y[ind]);
                        }
                        STEP = STEP * 0.999;
                        LOOP++;
                }
                end = omp_get_wtime();
         #if NO_FWRITE == 0
                cout << "Efficient V2 Minibatch Size:" << BATCHSIZE  << endl;
                cout << "Efficient V2 Minbatch Energy:" << ENERGY << endl;
                cout << "Efficient V2 Minibatch Parallel Wall time required:" << end - start << endl;
                writeToFile("EfficientV2"+ to_string(BATCHSIZE)+"PARAOUT" + to_string(LOOP));
         #endif
                result.push_back(ENERGY);
                result.push_back(end - start);
                return result;
        }
#endif

	void newalgo::print(){
		for(INDEXTYPE i = 0; i < graph.rows; i++){
                	cout << "Node:" << i << ", X:" << nCoordinates[i].getX() << ", Y:" << nCoordinates[i].getY()<< endl;
        	}
		cout << endl;
	}
	void newalgo::writeToFile(string f){
		stringstream  data(filename);
    		string lasttok;
    		while(getline(data,lasttok,'/'));
		filename = outputdir + lasttok + f + ".txt";
		ofstream output;
		output.open(filename);
		cout << "Creating output file in following directory:" << filename << endl;
		for(INDEXTYPE i = 0; i < graph.rows; i++){
			output << nCoordinates[i].getX() <<"\t"<< nCoordinates[i].getY() << "\t" << i+1 << endl;
		}
		output.close();
	}
	void newalgo::writeToFileEFF(string f){
		stringstream  data(filename);
                string lasttok;
                while(getline(data,lasttok,'/'));
                filename = outputdir + lasttok + f + ".txt";
                ofstream output;
                output.open(filename);
                cout << "Creating output file in following directory:" << filename << endl;
                for(INDEXTYPE i = 0; i < graph.rows; i++){
                        output << blasX[i] <<"\t"<< blasY[i] << "\t" << i+1 << endl;
                }
                output.close();
	}
/*
 *       Using intrinsic codes .... 
 *    
 */
/*
 * **************************** Intrinsic Version ************************ 
 */
#if 0
   vector<VALUETYPE> newalgo::EfficientVersionVec(INDEXTYPE ITERATIONS, 
         INDEXTYPE NUMOFTHREADS, INDEXTYPE BATCHSIZE)
   {
      INDEXTYPE LOOP = 0;
      VALUETYPE start, end, ENERGY, ENERGY0, *pb_X, *pb_Y;
      VALUETYPE STEP = 1.0;
      
      vector<VALUETYPE> result;
      pb_X = static_cast<VALUETYPE *> (::operator new (sizeof(VALUETYPE[BATCHSIZE])));
      pb_Y = static_cast<VALUETYPE *> (::operator new (sizeof(VALUETYPE[BATCHSIZE])));
      ENERGY0 = ENERGY = numeric_limits<VALUETYPE>::max();
     
#if 0
      cout << "Rows = " << graph.rows << endl;
      exit(1);
#endif

      omp_set_num_threads(NUMOFTHREADS);
/*
 *    time included initDFS 
 */
      start = omp_get_wtime();

      initDFS();
      
      while(LOOP < ITERATIONS)
      {
         ENERGY0 = ENERGY;
         ENERGY = 0;
         for(int i = 0; i < BATCHSIZE; i++)
         {
            pb_X[i] = pb_Y[i] = 0;
         }
	
         
	 for(INDEXTYPE b = 0; b < (graph.rows / BATCHSIZE); b += 1)
         {

/*
 *          NOTE: no cleanup for unroll as lon as BATCHZIE is multiple of unroll
 *          factor 
 */
	    #pragma omp parallel for schedule(static)	
	    for(INDEXTYPE i = b * BATCHSIZE; i < (b + 1) * BATCHSIZE; i += 8)
            {
	       int ind = i-b*BATCHSIZE;

	       register VTYPE vbx0, vbx1, vbx2, vbx3, vbx4, vbx5, vbx6, vbx7, dc;
               
               VALUETYPE distX, distY, dist; 
               VALUETYPE fx0;
               VALUETYPE fy0;
               
               
               VALUETYPE x0;
               VALUETYPE y0;
               VALUETYPE d0;
					
	       x0 = blasX[i];

	       y0 = blasY[i];
					
	       fx0 = 0;
	       fy0 = 0;		
		
/*
 *              j is up to i... lower up case  
 */
               for(INDEXTYPE j = 0; j < i; j += 1)
               {
                  VALUETYPE dx0, dx1;
                  VALUETYPE dy0, dy1;
                  VALUETYPE xj = blasX[j];
                  VALUETYPE yj = blasY[j];
		  
                  dx0 = xj - x0;
                  dy0 = yj - y0;
		
                  //distX = blasX[j] - blasX[i];
                  //distY = blasY[j] - blasY[i];
                                   	
	          d0 = 1.0 / (dx0 * dx0 + dy0 * dy0);
		
                  //dist2 = 1.0 / (distX * distX + distY * distY);
                                            
		  fx0 += dx0 * d0;
                  fy0 += dy0 * d0;
		 
                  //fx += distX * dist2;
                  //fy += distY * dist2;
               }		
/*
 *             location where we need to skip some points i==j
 *             NOTE: its UR iteration, no need to optimize 
 *             Hopefully compiler will unroll and get rid of the conditions
 */
/*
 *             Upper part  
 *
 */ 
               for(INDEXTYPE j = i+1; j < graph.rows; j += 1)
               {
                  VALUETYPE dx0, dx1;
                  VALUETYPE dy0, dy1;
                  VALUETYPE xj = blasX[j];
                  VALUETYPE yj = blasY[j];
		  
                  dx0 = xj - x0;
                  dy0 = yj - y0;
		
                  //distX = blasX[j] - blasX[i];
                  //distY = blasY[j] - blasY[i];
                                   	
	          d0 = 1.0 / (dx0 * dx0 + dy0 * dy0);
		
                  //dist2 = 1.0 / (distX * distX + distY * distY);
                                            
		  fx0 += dx0 * d0;
		
                  fy0 += dy0 * d0;
		 
                  //fx += distX * dist2;
                  //fy += distY * dist2;
               }
               
               
               // connected nodes 
               VALUETYPE px0=0, px1=0, px2=0, px3=0, px4=0, px5=0, px6=0, px7=0;
               VALUETYPE py0=0, py1=0, py2=0, py3=0, py4=0, py5=0, py6=0, py7=0;
               
               for(INDEXTYPE j = graph.rowptr[i]; j < graph.rowptr[i+1]; j += 1)
               {
                  int v = graph.colids[j];
                  distX = blasX[v] - x0;
                  distY = blasY[v] - y0;
                  dist = (distX * distX + distY * distY);
                  dist = sqrt(dist) + 1.0 / dist;

                  px0 += distX * dist;
                  py0 += distY * dist;
               }
				
               pb_X[ind] = px0 - fx0;
	       pb_Y[ind] = py0 - fy0;
	    }
	    /*printf("After:\n");
            for(INDEXTYPE i = b * BATCHSIZE; i < (b + 1) * BATCHSIZE; i++)
            {
               printf("%d = %lf,", i, pb_X[i-b*BATCHSIZE]);
            }
	    */
	    for(INDEXTYPE i = b * BATCHSIZE; i < (b + 1) * BATCHSIZE; i += 1)
            {
               int ind = i-b*BATCHSIZE;
               VALUETYPE d;
	       d = (pb_X[ind] * pb_X[ind] + pb_Y[ind] * pb_Y[ind]);
	       ENERGY += d ;
	       
               d = STEP / sqrt(d);
	       blasX[i] += pb_X[ind] * d;
               blasY[i] += pb_Y[ind] * d;
	       
               pb_X[ind] = 0;
	       pb_Y[ind] = 0;    
            }
         }
/*
 *    cleanup 
 */
			INDEXTYPE cleanup = (graph.rows/BATCHSIZE) * BATCHSIZE;
			#pragma omp parallel for schedule(dynamic) 
			for(INDEXTYPE i = cleanup; i < graph.rows; i += 1){
                                INDEXTYPE ind = i- cleanup;
                                VALUETYPE fx = 0, fy = 0, distX, distY, dist, dist2;
                                for(INDEXTYPE j = graph.rowptr[i]; j < graph.rowptr[i+1]; j += 1){
                                        int v = graph.colids[j];
                                        distX = blasX[v] - blasX[i];
                                        distY = blasY[v] - blasY[i];
                                        dist = (distX * distX + distY * distY);
                                        dist = sqrt(dist) + 1.0 / dist;
                                        pb_X[ind] += distX * dist;
                                        pb_Y[ind] += distY * dist;
                                }
                                for(INDEXTYPE j = 0; j < i; j += 1){
                                        distX = blasX[j] - blasX[i];
                                        distY = blasY[j] - blasY[i];
                                        dist2 = 1.0 / (distX * distX + distY * distY);
                                        fx += distX * dist2;
                                        fy += distY * dist2;
                                }
                                for(INDEXTYPE j = i+1; j < graph.rows; j += 1){
                                        distX = blasX[j] - blasX[i];
                                        distY = blasY[j] - blasY[i];
                                        dist2 = 1.0 / (distX * distX + distY * distY);
                                        fx += distX * dist2;
                                        fy += distY * dist2;
                                }
                                pb_X[ind] = pb_X[ind] - fx;
                                pb_Y[ind] = pb_Y[ind] - fy;
                        }	
			for(INDEXTYPE i = cleanup; i < graph.rows; i += 1){
                                int ind = i-cleanup;
                                double dist = (pb_X[ind]*pb_X[ind] + pb_Y[ind]*pb_Y[ind]);
                                ENERGY += dist;
				dist = STEP / sqrt(dist);
				blasX[i] += pb_X[ind] * dist;
                                blasY[i] += pb_Y[ind] * dist;
				pb_X[ind] = pb_Y[ind] = 0;
                        }
                        STEP = STEP * 0.999;
                        LOOP++;
                }
                end = omp_get_wtime();
         #if NO_FWRITE == 0
                cout << "Efficientunroll Minibatch Size:" << BATCHSIZE  << endl;
                cout << "Efficientunroll Minbatch Energy:" << ENERGY << endl;
                cout << "Efficientunroll Minibatch Parallel Wall time required:" << end - start << endl;
                writeToFileEFF("EFFUR"+ to_string(BATCHSIZE)+"PARAOUT" + to_string(LOOP));
         #endif
                result.push_back(ENERGY);
                result.push_back(end - start);
                return result;
        }
#endif

/*
 * **************   M-DIM Parallelizaion **************************************
 * NOTE: Since we have to split the loop .... using omp for would be costly
 * We must have to manage it manually..... 
 * We need to urnoll the outer loop as much as possible:  8*VLEN = 64 
 *    May need to write intrinsic code to make sure vectoried.. though compiler
 *    is not bad to vectorize following code
 * ToDO: manage it manually........ 
 */
#if 1 
   void PrintVector(VTYPE v1)
   {
      int i, vl=8; 
      double *fptr = (double*) &v1; 

      fprintf(stdout, "vector: < ");
      for(i=0; i < vl; i++)
         fprintf(stdout, "%lf, ",fptr[i] );
      fprintf(stdout, ">\n");
   }
#endif
   #ifdef MDIM_UR8 

   vector<VALUETYPE> newalgo::EfficientVersionMdim(INDEXTYPE ITERATIONS, 
         INDEXTYPE NUMOFTHREADS, INDEXTYPE BATCHSIZE)
   {
      INDEXTYPE LOOP = 0;
      VALUETYPE start, end, ENERGY, ENERGY0, *pb_X, *pb_Y;
      VALUETYPE STEP = 1.0;
      
      
      vector<VALUETYPE> result;
      pb_X = static_cast<VALUETYPE *> (::operator new (sizeof(VALUETYPE[BATCHSIZE])));
      pb_Y = static_cast<VALUETYPE *> (::operator new (sizeof(VALUETYPE[BATCHSIZE])));
      ENERGY0 = ENERGY = numeric_limits<VALUETYPE>::max();
     
      omp_set_num_threads(NUMOFTHREADS);
/*
 *    time included initDFS 
 */
      start = omp_get_wtime();

      initDFS();
      
      while(LOOP < ITERATIONS)
      {
         ENERGY0 = ENERGY;
         ENERGY = 0;
         for(int i = 0; i < BATCHSIZE; i++)
         {
            pb_X[i] = pb_Y[i] = 0;
         }
	
         
	 for(INDEXTYPE b = 0; b < (graph.rows / BATCHSIZE); b += 1)
         {
	    for(INDEXTYPE i = b * BATCHSIZE; i < (b + 1) * BATCHSIZE; i += 8)
            {
	       int ind = i-b*BATCHSIZE;

               VALUETYPE fx0, fx1, fx2, fx3, fx4, fx5, fx6, fx7;
               VALUETYPE fy0, fy1, fy2, fy3, fy4, fy5, fy6, fy7;
               
               VALUETYPE x0, x1, x2, x3, x4, x5, x6, x7;
               VALUETYPE y0, y1, y2, y3, y4, y5, y6, y7;
					
	       x0 = blasX[i];
	       x1 = blasX[i+1];
	       x2 = blasX[i+2];
	       x3 = blasX[i+3];
	       x4 = blasX[i+4];
	       x5 = blasX[i+5];
	       x6 = blasX[i+6];
	       x7 = blasX[i+7];

	       y0 = blasY[i];
	       y1 = blasY[i+1];	
	       y2 = blasY[i+2];
	       y3 = blasY[i+3];
	       y4 = blasY[i+4];
	       y5 = blasY[i+5];
	       y6 = blasY[i+6];
	       y7 = blasY[i+7];
					
	       fx0 = fx1 = fx2 = fx3 = fx4 = fx5 = fx6 = fx7 = 0;
	       fy0 = fy1 = fy2 = fy3 = fy4 = fy5 = fy6 = fy7 = 0;		
	
               #pragma omp parallel
               {
                  int id, nthreads; 
                  int chunksize;
                  int j, remainder; 
                  VALUETYPE dx0, dx1, dx2, dx3, dx4, dx5, dx6, dx7;
                  VALUETYPE dy0, dy1, dy2, dy3, dy4, dy5, dy6, dy7;
                  VALUETYPE d0, d1, d2, d3, d4, d5, d6, d7;
                  
                  VALUETYPE tfx0=0.0, tfx1=0.0, tfx2=0.0, tfx3=0.0, tfx4=0.0, 
                            tfx5=0.0, tfx6=0.0, tfx7=0.0;
                  VALUETYPE tfy0=0.0, tfy1=0.0, tfy2=0.0, tfy3=0.0, tfy4=0.0, 
                            tfy5=0.0, tfy6=0.0, tfy7=0.0;

                  const int M = graph.rows; 
                  id = omp_get_thread_num();
                  nthreads = omp_get_num_threads();

/*
 *                j is up to i... lower up case  
 */
                  chunksize = i/nthreads; 
                  for(j = id*chunksize; j < (id+1)*chunksize; j++)
                  {
                     VALUETYPE xj = blasX[j];
                     VALUETYPE yj = blasY[j];
		  
                     dx0 = xj - x0;
		     dx1 = xj - x1;
		     dx2 = xj - x2;
                     dx3 = xj - x3;
		     dx4 = xj - x4;
                     dx5 = xj - x5;
                     dx6 = xj - x6;
                     dx7 = xj - x7;
	          
                     dy0 = yj - y0;
		     dy1 = yj - y1;
		     dy2 = yj - y2;
                     dy3 = yj - y3;
		     dy4 = yj - y4;
                     dy5 = yj - y5;
                     dy6 = yj - y6;
                     dy7 = yj - y7; 	       
		
	             d0 = 1.0 / (dx0 * dx0 + dy0 * dy0);
		     d1 = 1.0 / (dx1 * dx1 + dy1 * dy1);
		     d2 = 1.0 / (dx2 * dx2 + dy2 * dy2);
		     d3 = 1.0 / (dx3 * dx3 + dy3 * dy3);
		     d4 = 1.0 / (dx4 * dx4 + dy4 * dy4);
		     d5 = 1.0 / (dx5 * dx5 + dy5 * dy5);
		     d6 = 1.0 / (dx6 * dx6 + dy6 * dy6);
		     d7 = 1.0 / (dx7 * dx7 + dy7 * dy7);       
		
		     tfx0 += dx0 * d0;
		     tfx1 += dx1 * d1;
		     tfx2 += dx2 * d2;
                     tfx3 += dx3 * d3;
		     tfx4 += dx4 * d4;
                     tfx5 += dx5 * d5;
                     tfx6 += dx6 * d6;
                     tfx7 += dx7 * d7;
		
                     tfy0 += dy0 * d0;
		     tfy1 += dy1 * d1;
		     tfy2 += dy2 * d2;
                     tfy3 += dy3 * d3;
		     tfy4 += dy4 * d4;
                     tfy5 += dy5 * d5;
                     tfy6 += dy6 * d6;
                     tfy7 += dy7 * d7;
                  }
/*
 *                cleanup of lower  
 */
                  #pragma omp single nowait
                  {
                     for (j=nthreads*chunksize; j < i; j++)
                     {
                        VALUETYPE xj = blasX[j];
                        VALUETYPE yj = blasY[j];
		  
                        dx0 = xj - x0;
		        dx1 = xj - x1;
		        dx2 = xj - x2;
                        dx3 = xj - x3;
		        dx4 = xj - x4;
                        dx5 = xj - x5;
                        dx6 = xj - x6;
                        dx7 = xj - x7;
	          
                        dy0 = yj - y0;
		        dy1 = yj - y1;
		        dy2 = yj - y2;
                        dy3 = yj - y3;
		        dy4 = yj - y4;
                        dy5 = yj - y5;
                        dy6 = yj - y6;
                        dy7 = yj - y7; 	       
		
	                d0 = 1.0 / (dx0 * dx0 + dy0 * dy0);
		        d1 = 1.0 / (dx1 * dx1 + dy1 * dy1);
		        d2 = 1.0 / (dx2 * dx2 + dy2 * dy2);
		        d3 = 1.0 / (dx3 * dx3 + dy3 * dy3);
		        d4 = 1.0 / (dx4 * dx4 + dy4 * dy4);
		        d5 = 1.0 / (dx5 * dx5 + dy5 * dy5);
		        d6 = 1.0 / (dx6 * dx6 + dy6 * dy6);
		        d7 = 1.0 / (dx7 * dx7 + dy7 * dy7);       
		
		        tfx0 += dx0 * d0;
		        tfx1 += dx1 * d1;
		        tfx2 += dx2 * d2;
                        tfx3 += dx3 * d3;
		        tfx4 += dx4 * d4;
                        tfx5 += dx5 * d5;
                        tfx6 += dx6 * d6;
                        tfx7 += dx7 * d7;
		
                        tfy0 += dy0 * d0;
		        tfy1 += dy1 * d1;
		        tfy2 += dy2 * d2;
                        tfy3 += dy3 * d3;
		        tfy4 += dy4 * d4;
                        tfy5 += dy5 * d5;
                        tfy6 += dy6 * d6;
                        tfy7 += dy7 * d7;
                     }
                  //}

/*
 *             location where we need to skip some points i==j
 *             NOTE: its UR iteration, no need to optimize 
 *             Hopefully compiler will unroll and get rid of the conditions
 */
                  //#pragma omp single nowait
                  //{
                     for (j = i; j < i+8; j++) 
                     {
                        VALUETYPE xj = blasX[j];
                        VALUETYPE yj = blasY[j];
                        if (j != i)
                        {
                           VALUETYPE dx0, dy0;
                           dx0 = xj - x0;
                           dy0 = yj - y0;
	                   d0 = 1.0 / (dx0 * dx0 + dy0 * dy0);
		           tfx0 += dx0 * d0;
                           tfy0 += dy0 * d0;
                        }
                        if ( j != i+1 )
                        {
                           VALUETYPE dx1, dy1;
                           dx1 = xj - x1;
                           dy1 = yj - y1;
	                   d1 = 1.0 / (dx1 * dx1 + dy1 * dy1);
		           tfx1 += dx1 * d1;
                           tfy1 += dy1 * d1;
                        }
                        if ( j != i+2 )
                        {
                           VALUETYPE dx2, dy2;
                           dx2 = xj - x2;
                           dy2 = yj - y2;
	                   d2 = 1.0 / (dx2 * dx2 + dy2 * dy2);
		           tfx2 += dx2 * d2;
                           tfy2 += dy2 * d2;
                        }
                        if ( j != i+3 )
                        {
                           VALUETYPE dx3, dy3;
                           dx3 = xj - x3;
                           dy3 = yj - y3;
	                   d3 = 1.0 / (dx3 * dx3 + dy3 * dy3);
		           tfx3 += dx3 * d3;
                           tfy3 += dy3 * d3;
                        }
                        if ( j != i+4 )
                        {
                           VALUETYPE dx4, dy4;
                           dx4 = xj - x4;
                           dy4 = yj - y4;
	                   d4 = 1.0 / (dx4 * dx4 + dy4 * dy4);
		           tfx4 += dx4 * d4;
                           tfy4 += dy4 * d4;
                        }
                        if ( j != i+5 )
                        {
                           VALUETYPE dx5, dy5;
                           dx5 = xj - x5;
                           dy5 = yj - y5;
	                   d5 = 1.0 / (dx5 * dx5 + dy5 * dy5);
		           tfx5 += dx5 * d5;
                           tfy5 += dy5 * d5;
                        }
                        if ( j != i+6 )
                        {
                           VALUETYPE dx6, dy6;
                           dx6 = xj - x6;
                           dy6 = yj - y6;
	                   d6 = 1.0 / (dx6 * dx6 + dy6 * dy6);
		           tfx6 += dx6 * d6;
                           tfy6 += dy6 * d6;
                        }
                        if ( j != i+7 )
                        {
                           VALUETYPE dx7, dy7;
                           dx7 = xj - x7;
                           dy7 = yj - y7;
	                   d7 = 1.0 / (dx7 * dx7 + dy7 * dy7);
		           tfx7 += dx7 * d7;
                           tfy7 += dy7 * d7;
                        }
                     }
                  }
/*
 *                Upper part  
 *
                  for(INDEXTYPE j = i+8; j < graph.rows; j += 1)
 */ 
                  chunksize = (M - i - 8)/nthreads; 
         #if DEBUG == 1
                  if (id == 0)
                  {
                     fprintf(DOUT, "upper half case: chunksize = %d\n", 
                           chunksize);
                  }
         #endif

                  for(j = (i+8)+id*chunksize; j < (i+8)+(id+1)*chunksize; j++)
                  {
                     VALUETYPE xj = blasX[j];
                     VALUETYPE yj = blasY[j];
		  
                     dx0 = xj - x0;
		     dx1 = xj - x1;
		     dx2 = xj - x2;
                     dx3 = xj - x3;
		     dx4 = xj - x4;
                     dx5 = xj - x5;
                     dx6 = xj - x6;
                     dx7 = xj - x7;
	          
                     dy0 = yj - y0;
		     dy1 = yj - y1;
		     dy2 = yj - y2;
                     dy3 = yj - y3;
		     dy4 = yj - y4;
                     dy5 = yj - y5;
                     dy6 = yj - y6;
                     dy7 = yj - y7; 	       
                                   	
	             d0 = 1.0 / (dx0 * dx0 + dy0 * dy0);
		     d1 = 1.0 / (dx1 * dx1 + dy1 * dy1);
		     d2 = 1.0 / (dx2 * dx2 + dy2 * dy2);
		     d3 = 1.0 / (dx3 * dx3 + dy3 * dy3);
		     d4 = 1.0 / (dx4 * dx4 + dy4 * dy4);
		     d5 = 1.0 / (dx5 * dx5 + dy5 * dy5);
		     d6 = 1.0 / (dx6 * dx6 + dy6 * dy6);
		     d7 = 1.0 / (dx7 * dx7 + dy7 * dy7);       
		
		     tfx0 += dx0 * d0;
		     tfx1 += dx1 * d1;
		     tfx2 += dx2 * d2;
                     tfx3 += dx3 * d3;
		     tfx4 += dx4 * d4;
                     tfx5 += dx5 * d5;
                     tfx6 += dx6 * d6;
                     tfx7 += dx7 * d7;
		
                     tfy0 += dy0 * d0;
		     tfy1 += dy1 * d1;
		     tfy2 += dy2 * d2;
                     tfy3 += dy3 * d3;
		     tfy4 += dy4 * d4;
                     tfy5 += dy5 * d5;
                     tfy6 += dy6 * d6;
                     tfy7 += dy7 * d7;
                  }
/*
 *                cleanup 
 */
                  #pragma omp single nowait 
                  {
         #if DEBUG == 1
                     fprintf(DOUT, "i=%d (id=%d): lower cleanup case\n", i, id); 
         #endif
                     for (j=(i+8)+nthreads*chunksize; j < M; j++)
                     {
                        VALUETYPE xj = blasX[j];
                        VALUETYPE yj = blasY[j];
		  
                        dx0 = xj - x0;
		        dx1 = xj - x1;
		        dx2 = xj - x2;
                        dx3 = xj - x3;
		        dx4 = xj - x4;
                        dx5 = xj - x5;
                        dx6 = xj - x6;
                        dx7 = xj - x7;
	          
                        dy0 = yj - y0;
		        dy1 = yj - y1;
		        dy2 = yj - y2;
                        dy3 = yj - y3;
		        dy4 = yj - y4;
                        dy5 = yj - y5;
                        dy6 = yj - y6;
                        dy7 = yj - y7; 	       
                                   	
	                d0 = 1.0 / (dx0 * dx0 + dy0 * dy0);
		        d1 = 1.0 / (dx1 * dx1 + dy1 * dy1);
		        d2 = 1.0 / (dx2 * dx2 + dy2 * dy2);
		        d3 = 1.0 / (dx3 * dx3 + dy3 * dy3);
		        d4 = 1.0 / (dx4 * dx4 + dy4 * dy4);
		        d5 = 1.0 / (dx5 * dx5 + dy5 * dy5);
		        d6 = 1.0 / (dx6 * dx6 + dy6 * dy6);
		        d7 = 1.0 / (dx7 * dx7 + dy7 * dy7);       
		
		        tfx0 += dx0 * d0;
		        tfx1 += dx1 * d1;
		        tfx2 += dx2 * d2;
                        tfx3 += dx3 * d3;
		        tfx4 += dx4 * d4;
                        tfx5 += dx5 * d5;
                        tfx6 += dx6 * d6;
                        tfx7 += dx7 * d7;
		
                        tfy0 += dy0 * d0;
		        tfy1 += dy1 * d1;
		        tfy2 += dy2 * d2;
                        tfy3 += dy3 * d3;
		        tfy4 += dy4 * d4;
                        tfy5 += dy5 * d5;
                        tfy6 += dy6 * d6;
                        tfy7 += dy7 * d7;
                     }
                  }
/*
 *                Reduction 
 */
                  #pragma omp critical
                  {
                     fx0 += tfx0; 
                     fx1 += tfx1; 
                     fx2 += tfx2; 
                     fx3 += tfx3; 
                     fx4 += tfx4; 
                     fx5 += tfx5; 
                     fx6 += tfx6; 
                     fx7 += tfx7; 
                     
                     fy0 += tfy0; 
                     fy1 += tfy1; 
                     fy2 += tfy2; 
                     fy3 += tfy3; 
                     fy4 += tfy4; 
                     fy5 += tfy5; 
                     fy6 += tfy6; 
                     fy7 += tfy7; 
                  }
               }
/*
 *             Out of parallel region 
 */
               pb_X[ind] = fx0;
	       pb_X[ind+1] = fx1;
	       pb_X[ind+2] = fx2;
               pb_X[ind+3] = fx3;
	       pb_X[ind+4] = fx4;
               pb_X[ind+5] = fx5;
               pb_X[ind+6] = fx6;
               pb_X[ind+7] = fx7;
					
	       pb_Y[ind] = fy0;
	       pb_Y[ind+1] = fy1;
	       pb_Y[ind+2] = fy2;
               pb_Y[ind+3] = fy3;
	       pb_Y[ind+4] = fy4;
               pb_Y[ind+5] = fy5;
               pb_Y[ind+6] = fy6;
               pb_Y[ind+7] = fy7;	
            }
            // connected nodes 
	    for(INDEXTYPE i = b * BATCHSIZE; i < (b + 1) * BATCHSIZE; i += 1)
            {   
               int ind = i-b*BATCHSIZE;
               VALUETYPE pbX=0.0, pbY=0.0;
               VALUETYPE dist, distX, distY;
               for(INDEXTYPE j = graph.rowptr[i]; j < graph.rowptr[i+1]; j += 1)
               {
                  int v = graph.colids[j];
                  distX = blasX[v] - blasX[i];
                  distY = blasY[v] - blasY[i];
                  dist = (distX * distX + distY * distY);
                  dist = sqrt(dist) + 1.0 / dist;
                  pbX += distX * dist;
                  pbY += distY * dist;
               }
               pb_X[ind] = pbX - pb_X[ind];
               pb_Y[ind] = pbY - pb_Y[ind];
            }
	    /*printf("After:\n");
            for(INDEXTYPE i = b * BATCHSIZE; i < (b + 1) * BATCHSIZE; i++)
            {
               printf("%d = %lf,", i, pb_X[i-b*BATCHSIZE]);
            }
	    */
	    for(INDEXTYPE i = b * BATCHSIZE; i < (b + 1) * BATCHSIZE; i += 1)
            {
               int ind = i-b*BATCHSIZE;
               VALUETYPE d;
	       d = (pb_X[ind] * pb_X[ind] + pb_Y[ind] * pb_Y[ind]);
	       ENERGY += d ;
	       
               d = STEP / sqrt(d);
	       blasX[i] += pb_X[ind] * d;
               blasY[i] += pb_Y[ind] * d;
	       
               pb_X[ind] = 0;
	       pb_Y[ind] = 0;    
            }
         }
/*
 *    cleanup 
 */
			INDEXTYPE cleanup = (graph.rows/BATCHSIZE) * BATCHSIZE;
			#pragma omp parallel for schedule(dynamic) 
			for(INDEXTYPE i = cleanup; i < graph.rows; i += 1){
                                INDEXTYPE ind = i- cleanup;
                                VALUETYPE fx = 0, fy = 0, distX, distY, dist, dist2;
                                for(INDEXTYPE j = graph.rowptr[i]; j < graph.rowptr[i+1]; j += 1){
                                        int v = graph.colids[j];
                                        distX = blasX[v] - blasX[i];
                                        distY = blasY[v] - blasY[i];
                                        dist = (distX * distX + distY * distY);
                                        dist = sqrt(dist) + 1.0 / dist;
                                        pb_X[ind] += distX * dist;
                                        pb_Y[ind] += distY * dist;
                                }
                                for(INDEXTYPE j = 0; j < i; j += 1){
                                        distX = blasX[j] - blasX[i];
                                        distY = blasY[j] - blasY[i];
                                        dist2 = 1.0 / (distX * distX + distY * distY);
                                        fx += distX * dist2;
                                        fy += distY * dist2;
                                }
                                for(INDEXTYPE j = i+1; j < graph.rows; j += 1){
                                        distX = blasX[j] - blasX[i];
                                        distY = blasY[j] - blasY[i];
                                        dist2 = 1.0 / (distX * distX + distY * distY);
                                        fx += distX * dist2;
                                        fy += distY * dist2;
                                }
                                pb_X[ind] = pb_X[ind] - fx;
                                pb_Y[ind] = pb_Y[ind] - fy;
                        }	
			for(INDEXTYPE i = cleanup; i < graph.rows; i += 1){
                                int ind = i-cleanup;
                                double dist = (pb_X[ind]*pb_X[ind] + pb_Y[ind]*pb_Y[ind]);
                                ENERGY += dist;
				dist = STEP / sqrt(dist);
				blasX[i] += pb_X[ind] * dist;
                                blasY[i] += pb_Y[ind] * dist;
				pb_X[ind] = pb_Y[ind] = 0;
                        }
                        STEP = STEP * 0.999;
                        LOOP++;
                }
                end = omp_get_wtime();
         
         #if NO_FWRITE == 0
                cout << "Efficientunroll Minibatch Size:" << BATCHSIZE  << endl;
                cout << "Efficientunroll Minbatch Energy:" << ENERGY << endl;
                cout << "Efficientunroll Minibatch Parallel Wall time required:" << end - start << endl;
                writeToFileEFF("EFFUR"+ to_string(BATCHSIZE)+"PARAOUT" + to_string(LOOP));
         #endif
                result.push_back(ENERGY);
                result.push_back(end - start);
                return result;
        }
/*
 *    Without critical section 
 */

#elif defined (MDIM_UR8_NOSYNC) 
   

   vector<VALUETYPE> newalgo::EfficientVersionMdim(INDEXTYPE ITERATIONS, 
         INDEXTYPE NUMOFTHREADS, INDEXTYPE BATCHSIZE)
   {
      INDEXTYPE LOOP = 0;
      VALUETYPE start, end, ENERGY, ENERGY0, *pb_X, *pb_Y;
      VALUETYPE STEP = 1.0;
      VALUETYPE *sumX, *sumY;
      
      
      vector<VALUETYPE> result;
      pb_X = static_cast<VALUETYPE *> (::operator new (sizeof(VALUETYPE[BATCHSIZE])));
      pb_Y = static_cast<VALUETYPE *> (::operator new (sizeof(VALUETYPE[BATCHSIZE])));
      ENERGY0 = ENERGY = numeric_limits<VALUETYPE>::max();
      
/*
 *    NOTE: to avoid overflow the stack, we allocate memory in heap to 
 *    manage reduction .... 
 *    FIXME: possible to combine it with pb_X and pb_Y???? 
 *           size = NUMOFTHREADS * UNROLL * VLEN
 *    solved the stack overflow by this way 
 *    NOTE: for single precision, it may suffer from false sharing: 4*8=32 byte
 */
      sumX = static_cast<VALUETYPE *> 
               (::operator new (sizeof(VALUETYPE[NUMOFTHREADS*8])));
      sumY = static_cast<VALUETYPE *> 
               (::operator new (sizeof(VALUETYPE[NUMOFTHREADS*8])));
     

      omp_set_num_threads(NUMOFTHREADS);
/*
 *    time included initDFS 
 */
      start = omp_get_wtime();

      initDFS();
      
      while(LOOP < ITERATIONS)
      {
         ENERGY0 = ENERGY;
         ENERGY = 0;
         for(int i = 0; i < BATCHSIZE; i++)
         {
            pb_X[i] = pb_Y[i] = 0;
         }
	
         
	 for(INDEXTYPE b = 0; b < (graph.rows / BATCHSIZE); b += 1)
         {
	    for(INDEXTYPE i = b * BATCHSIZE; i < (b + 1) * BATCHSIZE; i += 8)
            {
	       int ind = i-b*BATCHSIZE;
               int m, n;
               
               for (m=0; m < NUMOFTHREADS*8; m++)
                  sumX[m] = sumY[m] = 0.0;

               VALUETYPE fx0, fx1, fx2, fx3, fx4, fx5, fx6, fx7;
               VALUETYPE fy0, fy1, fy2, fy3, fy4, fy5, fy6, fy7;
               
               VALUETYPE x0, x1, x2, x3, x4, x5, x6, x7;
               VALUETYPE y0, y1, y2, y3, y4, y5, y6, y7;
					
	       x0 = blasX[i];
	       x1 = blasX[i+1];
	       x2 = blasX[i+2];
	       x3 = blasX[i+3];
	       x4 = blasX[i+4];
	       x5 = blasX[i+5];
	       x6 = blasX[i+6];
	       x7 = blasX[i+7];

	       y0 = blasY[i];
	       y1 = blasY[i+1];	
	       y2 = blasY[i+2];
	       y3 = blasY[i+3];
	       y4 = blasY[i+4];
	       y5 = blasY[i+5];
	       y6 = blasY[i+6];
	       y7 = blasY[i+7];
					
	       fx0 = fx1 = fx2 = fx3 = fx4 = fx5 = fx6 = fx7 = 0.0;
	       fy0 = fy1 = fy2 = fy3 = fy4 = fy5 = fy6 = fy7 = 0.0;		
               
               #pragma omp parallel
               {
                  int id, nthreads; 
                  int chunksize;
                  int j, remainder; 
                  VALUETYPE dx0, dx1, dx2, dx3, dx4, dx5, dx6, dx7;
                  VALUETYPE dy0, dy1, dy2, dy3, dy4, dy5, dy6, dy7;
                  VALUETYPE d0, d1, d2, d3, d4, d5, d6, d7;
                  
                  VALUETYPE tfx0=0.0, tfx1=0.0, tfx2=0.0, tfx3=0.0, tfx4=0.0, 
                            tfx5=0.0, tfx6=0.0, tfx7=0.0;
                  VALUETYPE tfy0=0.0, tfy1=0.0, tfy2=0.0, tfy3=0.0, tfy4=0.0, 
                            tfy5=0.0, tfy6=0.0, tfy7=0.0;

                  const int M = graph.rows; 
                  id = omp_get_thread_num();
                  nthreads = omp_get_num_threads();

/*
 *                j is up to i... lower up case  
 */
                  chunksize = i/nthreads; 
                  
                  for(j = id*chunksize; j < (id+1)*chunksize; j++)
                  {
                     VALUETYPE xj = blasX[j];
                     VALUETYPE yj = blasY[j];
		  
                     dx0 = xj - x0;
		     dx1 = xj - x1;
		     dx2 = xj - x2;
                     dx3 = xj - x3;
		     dx4 = xj - x4;
                     dx5 = xj - x5;
                     dx6 = xj - x6;
                     dx7 = xj - x7;
	          
                     dy0 = yj - y0;
		     dy1 = yj - y1;
		     dy2 = yj - y2;
                     dy3 = yj - y3;
		     dy4 = yj - y4;
                     dy5 = yj - y5;
                     dy6 = yj - y6;
                     dy7 = yj - y7; 	       
		
	             d0 = 1.0 / (dx0 * dx0 + dy0 * dy0);
		     d1 = 1.0 / (dx1 * dx1 + dy1 * dy1);
		     d2 = 1.0 / (dx2 * dx2 + dy2 * dy2);
		     d3 = 1.0 / (dx3 * dx3 + dy3 * dy3);
		     d4 = 1.0 / (dx4 * dx4 + dy4 * dy4);
		     d5 = 1.0 / (dx5 * dx5 + dy5 * dy5);
		     d6 = 1.0 / (dx6 * dx6 + dy6 * dy6);
		     d7 = 1.0 / (dx7 * dx7 + dy7 * dy7);       
		
		     tfx0 += dx0 * d0;
		     tfx1 += dx1 * d1;
		     tfx2 += dx2 * d2;
                     tfx3 += dx3 * d3;
		     tfx4 += dx4 * d4;
                     tfx5 += dx5 * d5;
                     tfx6 += dx6 * d6;
                     tfx7 += dx7 * d7;
		
                     tfy0 += dy0 * d0;
		     tfy1 += dy1 * d1;
		     tfy2 += dy2 * d2;
                     tfy3 += dy3 * d3;
		     tfy4 += dy4 * d4;
                     tfy5 += dy5 * d5;
                     tfy6 += dy6 * d6;
                     tfy7 += dy7 * d7;
                  }
/*
 *                cleanup of lower  
 */
                  #pragma omp single nowait
                  {
                     for (j=nthreads*chunksize; j < i; j++)
                     {
                        VALUETYPE xj = blasX[j];
                        VALUETYPE yj = blasY[j];
		  
                        dx0 = xj - x0;
		        dx1 = xj - x1;
		        dx2 = xj - x2;
                        dx3 = xj - x3;
		        dx4 = xj - x4;
                        dx5 = xj - x5;
                        dx6 = xj - x6;
                        dx7 = xj - x7;
	          
                        dy0 = yj - y0;
		        dy1 = yj - y1;
		        dy2 = yj - y2;
                        dy3 = yj - y3;
		        dy4 = yj - y4;
                        dy5 = yj - y5;
                        dy6 = yj - y6;
                        dy7 = yj - y7; 	       
		
	                d0 = 1.0 / (dx0 * dx0 + dy0 * dy0);
		        d1 = 1.0 / (dx1 * dx1 + dy1 * dy1);
		        d2 = 1.0 / (dx2 * dx2 + dy2 * dy2);
		        d3 = 1.0 / (dx3 * dx3 + dy3 * dy3);
		        d4 = 1.0 / (dx4 * dx4 + dy4 * dy4);
		        d5 = 1.0 / (dx5 * dx5 + dy5 * dy5);
		        d6 = 1.0 / (dx6 * dx6 + dy6 * dy6);
		        d7 = 1.0 / (dx7 * dx7 + dy7 * dy7);       
		
		        tfx0 += dx0 * d0;
		        tfx1 += dx1 * d1;
		        tfx2 += dx2 * d2;
                        tfx3 += dx3 * d3;
		        tfx4 += dx4 * d4;
                        tfx5 += dx5 * d5;
                        tfx6 += dx6 * d6;
                        tfx7 += dx7 * d7;
		
                        tfy0 += dy0 * d0;
		        tfy1 += dy1 * d1;
		        tfy2 += dy2 * d2;
                        tfy3 += dy3 * d3;
		        tfy4 += dy4 * d4;
                        tfy5 += dy5 * d5;
                        tfy6 += dy6 * d6;
                        tfy7 += dy7 * d7;
                     }
                  //}

/*
 *             location where we need to skip some points i==j
 *             NOTE: its UR iteration, no need to optimize 
 *             Hopefully compiler will unroll and get rid of the conditions
 */
                  //#pragma omp single nowait
                  //{
                     for (j = i; j < i+8; j++) 
                     {
                        VALUETYPE xj = blasX[j];
                        VALUETYPE yj = blasY[j];
                        if (j != i)
                        {
                           VALUETYPE dx0, dy0;
                           dx0 = xj - x0;
                           dy0 = yj - y0;
	                   d0 = 1.0 / (dx0 * dx0 + dy0 * dy0);
		           tfx0 += dx0 * d0;
                           tfy0 += dy0 * d0;
                        }
                        if ( j != i+1 )
                        {
                           VALUETYPE dx1, dy1;
                           dx1 = xj - x1;
                           dy1 = yj - y1;
	                   d1 = 1.0 / (dx1 * dx1 + dy1 * dy1);
		           tfx1 += dx1 * d1;
                           tfy1 += dy1 * d1;
                        }
                        if ( j != i+2 )
                        {
                           VALUETYPE dx2, dy2;
                           dx2 = xj - x2;
                           dy2 = yj - y2;
	                   d2 = 1.0 / (dx2 * dx2 + dy2 * dy2);
		           tfx2 += dx2 * d2;
                           tfy2 += dy2 * d2;
                        }
                        if ( j != i+3 )
                        {
                           VALUETYPE dx3, dy3;
                           dx3 = xj - x3;
                           dy3 = yj - y3;
	                   d3 = 1.0 / (dx3 * dx3 + dy3 * dy3);
		           tfx3 += dx3 * d3;
                           tfy3 += dy3 * d3;
                        }
                        if ( j != i+4 )
                        {
                           VALUETYPE dx4, dy4;
                           dx4 = xj - x4;
                           dy4 = yj - y4;
	                   d4 = 1.0 / (dx4 * dx4 + dy4 * dy4);
		           tfx4 += dx4 * d4;
                           tfy4 += dy4 * d4;
                        }
                        if ( j != i+5 )
                        {
                           VALUETYPE dx5, dy5;
                           dx5 = xj - x5;
                           dy5 = yj - y5;
	                   d5 = 1.0 / (dx5 * dx5 + dy5 * dy5);
		           tfx5 += dx5 * d5;
                           tfy5 += dy5 * d5;
                        }
                        if ( j != i+6 )
                        {
                           VALUETYPE dx6, dy6;
                           dx6 = xj - x6;
                           dy6 = yj - y6;
	                   d6 = 1.0 / (dx6 * dx6 + dy6 * dy6);
		           tfx6 += dx6 * d6;
                           tfy6 += dy6 * d6;
                        }
                        if ( j != i+7 )
                        {
                           VALUETYPE dx7, dy7;
                           dx7 = xj - x7;
                           dy7 = yj - y7;
	                   d7 = 1.0 / (dx7 * dx7 + dy7 * dy7);
		           tfx7 += dx7 * d7;
                           tfy7 += dy7 * d7;
                        }
                     }
                  }
/*
 *                Upper part  
 *
                  for(INDEXTYPE j = i+8; j < graph.rows; j += 1)
 */ 
                  chunksize = (M - i - 8)/nthreads; 

                  for(j = (i+8)+id*chunksize; j < (i+8)+(id+1)*chunksize; j++)
                  {
                     VALUETYPE xj = blasX[j];
                     VALUETYPE yj = blasY[j];
		  
                     dx0 = xj - x0;
		     dx1 = xj - x1;
		     dx2 = xj - x2;
                     dx3 = xj - x3;
		     dx4 = xj - x4;
                     dx5 = xj - x5;
                     dx6 = xj - x6;
                     dx7 = xj - x7;
	          
                     dy0 = yj - y0;
		     dy1 = yj - y1;
		     dy2 = yj - y2;
                     dy3 = yj - y3;
		     dy4 = yj - y4;
                     dy5 = yj - y5;
                     dy6 = yj - y6;
                     dy7 = yj - y7; 	       
                                   	
	             d0 = 1.0 / (dx0 * dx0 + dy0 * dy0);
		     d1 = 1.0 / (dx1 * dx1 + dy1 * dy1);
		     d2 = 1.0 / (dx2 * dx2 + dy2 * dy2);
		     d3 = 1.0 / (dx3 * dx3 + dy3 * dy3);
		     d4 = 1.0 / (dx4 * dx4 + dy4 * dy4);
		     d5 = 1.0 / (dx5 * dx5 + dy5 * dy5);
		     d6 = 1.0 / (dx6 * dx6 + dy6 * dy6);
		     d7 = 1.0 / (dx7 * dx7 + dy7 * dy7);       
		
		     tfx0 += dx0 * d0;
		     tfx1 += dx1 * d1;
		     tfx2 += dx2 * d2;
                     tfx3 += dx3 * d3;
		     tfx4 += dx4 * d4;
                     tfx5 += dx5 * d5;
                     tfx6 += dx6 * d6;
                     tfx7 += dx7 * d7;
		
                     tfy0 += dy0 * d0;
		     tfy1 += dy1 * d1;
		     tfy2 += dy2 * d2;
                     tfy3 += dy3 * d3;
		     tfy4 += dy4 * d4;
                     tfy5 += dy5 * d5;
                     tfy6 += dy6 * d6;
                     tfy7 += dy7 * d7;
                  }
/*
 *                cleanup 
 */
                  #pragma omp single nowait 
                  {
                     for (j=(i+8)+nthreads*chunksize; j < M; j++)
                     {
                        VALUETYPE xj = blasX[j];
                        VALUETYPE yj = blasY[j];
		  
                        dx0 = xj - x0;
		        dx1 = xj - x1;
		        dx2 = xj - x2;
                        dx3 = xj - x3;
		        dx4 = xj - x4;
                        dx5 = xj - x5;
                        dx6 = xj - x6;
                        dx7 = xj - x7;
	          
                        dy0 = yj - y0;
		        dy1 = yj - y1;
		        dy2 = yj - y2;
                        dy3 = yj - y3;
		        dy4 = yj - y4;
                        dy5 = yj - y5;
                        dy6 = yj - y6;
                        dy7 = yj - y7; 	       
                                   	
	                d0 = 1.0 / (dx0 * dx0 + dy0 * dy0);
		        d1 = 1.0 / (dx1 * dx1 + dy1 * dy1);
		        d2 = 1.0 / (dx2 * dx2 + dy2 * dy2);
		        d3 = 1.0 / (dx3 * dx3 + dy3 * dy3);
		        d4 = 1.0 / (dx4 * dx4 + dy4 * dy4);
		        d5 = 1.0 / (dx5 * dx5 + dy5 * dy5);
		        d6 = 1.0 / (dx6 * dx6 + dy6 * dy6);
		        d7 = 1.0 / (dx7 * dx7 + dy7 * dy7);       
		
		        tfx0 += dx0 * d0;
		        tfx1 += dx1 * d1;
		        tfx2 += dx2 * d2;
                        tfx3 += dx3 * d3;
		        tfx4 += dx4 * d4;
                        tfx5 += dx5 * d5;
                        tfx6 += dx6 * d6;
                        tfx7 += dx7 * d7;
		
                        tfy0 += dy0 * d0;
		        tfy1 += dy1 * d1;
		        tfy2 += dy2 * d2;
                        tfy3 += dy3 * d3;
		        tfy4 += dy4 * d4;
                        tfy5 += dy5 * d5;
                        tfy6 += dy6 * d6;
                        tfy7 += dy7 * d7;
                     }
                  }
/*
 *                Reduction using memory  
 */
                  {
                     sumX[id] += tfx0; 
                     sumX[id+1] += tfx1; 
                     sumX[id+2] += tfx2; 
                     sumX[id+3] += tfx3; 
                     sumX[id+4] += tfx4; 
                     sumX[id+5] += tfx5; 
                     sumX[id+6] += tfx6; 
                     sumX[id+7] += tfx7; 
                     
                     sumY[id] += tfy0; 
                     sumY[id+1] += tfy1; 
                     sumY[id+2] += tfy2; 
                     sumY[id+3] += tfy3; 
                     sumY[id+4] += tfy4; 
                     sumY[id+5] += tfy5; 
                     sumY[id+6] += tfy6; 
                     sumY[id+7] += tfy7; 
                  }
               }
/*
 *             Out of parallel region 
 */
/*
 *             sumup all the sums 
 */
               for (m=0; m < NUMOFTHREADS; m++)
               {
                  fx0 += sumX[m]; 
                  fx1 += sumX[m+1]; 
                  fx2 += sumX[m+2]; 
                  fx3 += sumX[m+3]; 
                  fx4 += sumX[m+4]; 
                  fx5 += sumX[m+5]; 
                  fx6 += sumX[m+6]; 
                  fx7 += sumX[m+7]; 
                  
                  fy0 += sumY[m]; 
                  fy1 += sumY[m+1]; 
                  fy2 += sumY[m+2]; 
                  fy3 += sumY[m+3]; 
                  fy4 += sumY[m+4]; 
                  fy5 += sumY[m+5]; 
                  fy6 += sumY[m+6]; 
                  fy7 += sumY[m+7]; 
               }


               pb_X[ind] = fx0;
	       pb_X[ind+1] = fx1;
	       pb_X[ind+2] = fx2;
               pb_X[ind+3] = fx3;
	       pb_X[ind+4] = fx4;
               pb_X[ind+5] = fx5;
               pb_X[ind+6] = fx6;
               pb_X[ind+7] = fx7;
					
	       pb_Y[ind] = fy0;
	       pb_Y[ind+1] = fy1;
	       pb_Y[ind+2] = fy2;
               pb_Y[ind+3] = fy3;
	       pb_Y[ind+4] = fy4;
               pb_Y[ind+5] = fy5;
               pb_Y[ind+6] = fy6;
               pb_Y[ind+7] = fy7;	
            }
            // connected nodes 
	    for(INDEXTYPE i = b * BATCHSIZE; i < (b + 1) * BATCHSIZE; i += 1)
            {   
               int ind = i-b*BATCHSIZE;
               VALUETYPE pbX=0.0, pbY=0.0;
               VALUETYPE dist, distX, distY;
               for(INDEXTYPE j = graph.rowptr[i]; j < graph.rowptr[i+1]; j += 1)
               {
                  int v = graph.colids[j];
                  distX = blasX[v] - blasX[i];
                  distY = blasY[v] - blasY[i];
                  dist = (distX * distX + distY * distY);
                  dist = sqrt(dist) + 1.0 / dist;
                  pbX += distX * dist;
                  pbY += distY * dist;
               }
               pb_X[ind] = pbX - pb_X[ind];
               pb_Y[ind] = pbY - pb_Y[ind];
            }
	    /*printf("After:\n");
            for(INDEXTYPE i = b * BATCHSIZE; i < (b + 1) * BATCHSIZE; i++)
            {
               printf("%d = %lf,", i, pb_X[i-b*BATCHSIZE]);
            }
	    */
	    for(INDEXTYPE i = b * BATCHSIZE; i < (b + 1) * BATCHSIZE; i += 1)
            {
               int ind = i-b*BATCHSIZE;
               VALUETYPE d;
	       d = (pb_X[ind] * pb_X[ind] + pb_Y[ind] * pb_Y[ind]);
	       ENERGY += d ;
	       
               d = STEP / sqrt(d);
	       blasX[i] += pb_X[ind] * d;
               blasY[i] += pb_Y[ind] * d;
	       
               pb_X[ind] = 0;
	       pb_Y[ind] = 0;    
            }
         }
/*
 *    cleanup 
 */
			INDEXTYPE cleanup = (graph.rows/BATCHSIZE) * BATCHSIZE;
			#pragma omp parallel for schedule(dynamic) 
			for(INDEXTYPE i = cleanup; i < graph.rows; i += 1){
                                INDEXTYPE ind = i- cleanup;
                                VALUETYPE fx = 0, fy = 0, distX, distY, dist, dist2;
                                for(INDEXTYPE j = graph.rowptr[i]; j < graph.rowptr[i+1]; j += 1){
                                        int v = graph.colids[j];
                                        distX = blasX[v] - blasX[i];
                                        distY = blasY[v] - blasY[i];
                                        dist = (distX * distX + distY * distY);
                                        dist = sqrt(dist) + 1.0 / dist;
                                        pb_X[ind] += distX * dist;
                                        pb_Y[ind] += distY * dist;
                                }
                                for(INDEXTYPE j = 0; j < i; j += 1){
                                        distX = blasX[j] - blasX[i];
                                        distY = blasY[j] - blasY[i];
                                        dist2 = 1.0 / (distX * distX + distY * distY);
                                        fx += distX * dist2;
                                        fy += distY * dist2;
                                }
                                for(INDEXTYPE j = i+1; j < graph.rows; j += 1){
                                        distX = blasX[j] - blasX[i];
                                        distY = blasY[j] - blasY[i];
                                        dist2 = 1.0 / (distX * distX + distY * distY);
                                        fx += distX * dist2;
                                        fy += distY * dist2;
                                }
                                pb_X[ind] = pb_X[ind] - fx;
                                pb_Y[ind] = pb_Y[ind] - fy;
                        }	
			for(INDEXTYPE i = cleanup; i < graph.rows; i += 1){
                                int ind = i-cleanup;
                                double dist = (pb_X[ind]*pb_X[ind] + pb_Y[ind]*pb_Y[ind]);
                                ENERGY += dist;
				dist = STEP / sqrt(dist);
				blasX[i] += pb_X[ind] * dist;
                                blasY[i] += pb_Y[ind] * dist;
				pb_X[ind] = pb_Y[ind] = 0;
                        }
                        STEP = STEP * 0.999;
                        LOOP++;
                }
                end = omp_get_wtime();
         #if NO_FWRITE == 0
                cout << "Efficientunroll Minibatch Size:" << BATCHSIZE  << endl;
                cout << "Efficientunroll Minbatch Energy:" << ENERGY << endl;
                cout << "Efficientunroll Minibatch Parallel Wall time required:" << end - start << endl;
                writeToFileEFF("EFFUR8-NOSYNC"+ to_string(BATCHSIZE)+"PARAOUT" + to_string(LOOP));
         #endif
                result.push_back(ENERGY);
                result.push_back(end - start);
                return result;
        }


#elif defined (MDIM_UR4) 

   vector<VALUETYPE> newalgo::EfficientVersionMdim(INDEXTYPE ITERATIONS, 
         INDEXTYPE NUMOFTHREADS, INDEXTYPE BATCHSIZE)
   {
      INDEXTYPE LOOP = 0;
      VALUETYPE start, end, ENERGY, ENERGY0, *pb_X, *pb_Y;
      VALUETYPE STEP = 1.0;
      
      
      vector<VALUETYPE> result;
      pb_X = static_cast<VALUETYPE *> (::operator new (sizeof(VALUETYPE[BATCHSIZE])));
      pb_Y = static_cast<VALUETYPE *> (::operator new (sizeof(VALUETYPE[BATCHSIZE])));
      ENERGY0 = ENERGY = numeric_limits<VALUETYPE>::max();
   

      omp_set_num_threads(NUMOFTHREADS);
/*
 *    time included initDFS 
 */
      start = omp_get_wtime();

      initDFS();
      
      while(LOOP < ITERATIONS)
      {
         ENERGY0 = ENERGY;
         ENERGY = 0;
         for(int i = 0; i < BATCHSIZE; i++)
         {
            pb_X[i] = pb_Y[i] = 0;
         }
	
         
	 for(INDEXTYPE b = 0; b < (graph.rows / BATCHSIZE); b += 1)
         {
	    for(INDEXTYPE i = b * BATCHSIZE; i < (b + 1) * BATCHSIZE; i += 4)
            {
	       int ind = i-b*BATCHSIZE;

               VALUETYPE fx0, fx1, fx2, fx3;
               VALUETYPE fy0, fy1, fy2, fy3;
               
               VALUETYPE x0, x1, x2, x3, x4;
               VALUETYPE y0, y1, y2, y3, y4;
					
	       x0 = blasX[i];
	       x1 = blasX[i+1];
	       x2 = blasX[i+2];
	       x3 = blasX[i+3];

	       y0 = blasY[i];
	       y1 = blasY[i+1];	
	       y2 = blasY[i+2];
	       y3 = blasY[i+3];
					
	       fx0 = fx1 = fx2 = fx3 = 0.0;
	       fy0 = fy1 = fy2 = fy3 = 0.0;		
	
               #pragma omp parallel
               {
                  int id, nthreads; 
                  int chunksize;
                  int j, remainder; 
                  VALUETYPE dx0, dx1, dx2, dx3;
                  VALUETYPE dy0, dy1, dy2, dy3;
                  VALUETYPE d0, d1, d2, d3;
                  
                  VALUETYPE tfx0=0.0, tfx1=0.0, tfx2=0.0, tfx3=0.0;
                  VALUETYPE tfy0=0.0, tfy1=0.0, tfy2=0.0, tfy3=0.0;

                  const int M = graph.rows; 
                  id = omp_get_thread_num();
                  nthreads = omp_get_num_threads();

/*
 *                j is up to i... lower up case  
 */
                  chunksize = i/nthreads; 
#if DEBUG == 1
                  if (id == 0)
                  {
                     fprintf(DOUT, "**********\n");
                     fprintf(DOUT, "lower half case: chunksize = %ld\n", 
                           chunksize );
                     fflush(DOUT);
                  }
#endif
                  for(j = id*chunksize; j < (id+1)*chunksize; j++)
                  {
                     VALUETYPE xj = blasX[j];
                     VALUETYPE yj = blasY[j];
		  
                     dx0 = xj - x0;
		     dx1 = xj - x1;
		     dx2 = xj - x2;
                     dx3 = xj - x3;
	          
                     dy0 = yj - y0;
		     dy1 = yj - y1;
		     dy2 = yj - y2;
                     dy3 = yj - y3;
		
	             d0 = 1.0 / (dx0 * dx0 + dy0 * dy0);
		     d1 = 1.0 / (dx1 * dx1 + dy1 * dy1);
		     d2 = 1.0 / (dx2 * dx2 + dy2 * dy2);
		     d3 = 1.0 / (dx3 * dx3 + dy3 * dy3);
		
		     tfx0 += dx0 * d0;
		     tfx1 += dx1 * d1;
		     tfx2 += dx2 * d2;
                     tfx3 += dx3 * d3;
		
                     tfy0 += dy0 * d0;
		     tfy1 += dy1 * d1;
		     tfy2 += dy2 * d2;
                     tfy3 += dy3 * d3;
                  }
/*
 *                cleanup of lower  
 */
                  #pragma omp single nowait
                  {
         #if DEBUG == 1
                     fprintf(DOUT, "i=%d: lower cleanup case = %d\n", i, 
                           i-nthreads*chunksize); 
         #endif
                     for (j=nthreads*chunksize; j < i; j++)
                     {
                        VALUETYPE xj = blasX[j];
                        VALUETYPE yj = blasY[j];
		  
                        dx0 = xj - x0;
		        dx1 = xj - x1;
		        dx2 = xj - x2;
                        dx3 = xj - x3;
	          
                        dy0 = yj - y0;
		        dy1 = yj - y1;
		        dy2 = yj - y2;
                        dy3 = yj - y3;
		
	                d0 = 1.0 / (dx0 * dx0 + dy0 * dy0);
		        d1 = 1.0 / (dx1 * dx1 + dy1 * dy1);
		        d2 = 1.0 / (dx2 * dx2 + dy2 * dy2);
		        d3 = 1.0 / (dx3 * dx3 + dy3 * dy3);
		
		        tfx0 += dx0 * d0;
		        tfx1 += dx1 * d1;
		        tfx2 += dx2 * d2;
                        tfx3 += dx3 * d3;
		
                        tfy0 += dy0 * d0;
		        tfy1 += dy1 * d1;
		        tfy2 += dy2 * d2;
                        tfy3 += dy3 * d3;
                     }
/*
 *             location where we need to skip some points i==j
 *             NOTE: its UR iteration, no need to optimize 
 *             Hopefully compiler will unroll and get rid of the conditions
 */
                     for (j = i; j < i+4; j++) 
                     {
                        VALUETYPE xj = blasX[j];
                        VALUETYPE yj = blasY[j];
                        if (j != i)
                        {
                           VALUETYPE dx0, dy0;
                           dx0 = xj - x0;
                           dy0 = yj - y0;
	                   d0 = 1.0 / (dx0 * dx0 + dy0 * dy0);
		           tfx0 += dx0 * d0;
                           tfy0 += dy0 * d0;
                        }
                        if ( j != i+1 )
                        {
                           VALUETYPE dx1, dy1;
                           dx1 = xj - x1;
                           dy1 = yj - y1;
	                   d1 = 1.0 / (dx1 * dx1 + dy1 * dy1);
		           tfx1 += dx1 * d1;
                           tfy1 += dy1 * d1;
                        }
                        if ( j != i+2 )
                        {
                           VALUETYPE dx2, dy2;
                           dx2 = xj - x2;
                           dy2 = yj - y2;
	                   d2 = 1.0 / (dx2 * dx2 + dy2 * dy2);
		           tfx2 += dx2 * d2;
                           tfy2 += dy2 * d2;
                        }
                        if ( j != i+3 )
                        {
                           VALUETYPE dx3, dy3;
                           dx3 = xj - x3;
                           dy3 = yj - y3;
	                   d3 = 1.0 / (dx3 * dx3 + dy3 * dy3);
		           tfx3 += dx3 * d3;
                           tfy3 += dy3 * d3;
                        }
                     }
                  }
/*
 *                Upper part  
 *
                  for(INDEXTYPE j = i+8; j < graph.rows; j += 1)
 */ 
                  chunksize = (M - i - 4)/nthreads; 
         #if DEBUG == 1
                  if (id == 0)
                  {
                     fprintf(DOUT, "upper half case: chunksize = %d\n", 
                           chunksize);
                  }
         #endif

                  for(j = (i+4)+id*chunksize; j < (i+4)+(id+1)*chunksize; j++)
                  {
                     VALUETYPE xj = blasX[j];
                     VALUETYPE yj = blasY[j];
		  
                     dx0 = xj - x0;
		     dx1 = xj - x1;
		     dx2 = xj - x2;
                     dx3 = xj - x3;
	          
                     dy0 = yj - y0;
		     dy1 = yj - y1;
		     dy2 = yj - y2;
                     dy3 = yj - y3;
                                   	
	             d0 = 1.0 / (dx0 * dx0 + dy0 * dy0);
		     d1 = 1.0 / (dx1 * dx1 + dy1 * dy1);
		     d2 = 1.0 / (dx2 * dx2 + dy2 * dy2);
		     d3 = 1.0 / (dx3 * dx3 + dy3 * dy3);
		
		     tfx0 += dx0 * d0;
		     tfx1 += dx1 * d1;
		     tfx2 += dx2 * d2;
                     tfx3 += dx3 * d3;
		
                     tfy0 += dy0 * d0;
		     tfy1 += dy1 * d1;
		     tfy2 += dy2 * d2;
                     tfy3 += dy3 * d3;
                  }
/*
 *                cleanup 
 */
                  #pragma omp single nowait 
                  {
         #if DEBUG == 1
                     fprintf(DOUT, "i=%d (id=%d): lower cleanup case\n", i, id); 
         #endif
                     for (j=(i+4)+nthreads*chunksize; j < M; j++)
                     {
                        VALUETYPE xj = blasX[j];
                        VALUETYPE yj = blasY[j];
		  
                        dx0 = xj - x0;
		        dx1 = xj - x1;
		        dx2 = xj - x2;
                        dx3 = xj - x3;
	          
                        dy0 = yj - y0;
		        dy1 = yj - y1;
		        dy2 = yj - y2;
                        dy3 = yj - y3;
                                   	
	                d0 = 1.0 / (dx0 * dx0 + dy0 * dy0);
		        d1 = 1.0 / (dx1 * dx1 + dy1 * dy1);
		        d2 = 1.0 / (dx2 * dx2 + dy2 * dy2);
		        d3 = 1.0 / (dx3 * dx3 + dy3 * dy3);
		
		        tfx0 += dx0 * d0;
		        tfx1 += dx1 * d1;
		        tfx2 += dx2 * d2;
                        tfx3 += dx3 * d3;
		
                        tfy0 += dy0 * d0;
		        tfy1 += dy1 * d1;
		        tfy2 += dy2 * d2;
                        tfy3 += dy3 * d3;
                     }
                  }
/*
 *                Reduction 
 */
                  #pragma omp critical
                  {
                     fx0 += tfx0; 
                     fx1 += tfx1; 
                     fx2 += tfx2; 
                     fx3 += tfx3; 
                     
                     fy0 += tfy0; 
                     fy1 += tfy1; 
                     fy2 += tfy2; 
                     fy3 += tfy3; 
                  }
               }
/*
 *             Out of parallel region 
 */
               pb_X[ind] = fx0;
	       pb_X[ind+1] = fx1;
	       pb_X[ind+2] = fx2;
               pb_X[ind+3] = fx3;
					
	       pb_Y[ind] = fy0;
	       pb_Y[ind+1] = fy1;
	       pb_Y[ind+2] = fy2;
               pb_Y[ind+3] = fy3;
            }
/*
 *          =============================================
 */
            // connected nodes 
	    for(INDEXTYPE i = b * BATCHSIZE; i < (b + 1) * BATCHSIZE; i += 1)
            {   
               int ind = i-b*BATCHSIZE;
               VALUETYPE pbX=0.0, pbY=0.0;
               VALUETYPE dist, distX, distY;
               for(INDEXTYPE j = graph.rowptr[i]; j < graph.rowptr[i+1]; j += 1)
               {
                  int v = graph.colids[j];
                  distX = blasX[v] - blasX[i];
                  distY = blasY[v] - blasY[i];
                  dist = (distX * distX + distY * distY);
                  dist = sqrt(dist) + 1.0 / dist;
                  pbX += distX * dist;
                  pbY += distY * dist;
               }
               pb_X[ind] = pbX - pb_X[ind];
               pb_Y[ind] = pbY - pb_Y[ind];
            }
	    /*printf("After:\n");
            for(INDEXTYPE i = b * BATCHSIZE; i < (b + 1) * BATCHSIZE; i++)
            {
               printf("%d = %lf,", i, pb_X[i-b*BATCHSIZE]);
            }
	    */
	    for(INDEXTYPE i = b * BATCHSIZE; i < (b + 1) * BATCHSIZE; i += 1)
            {
               int ind = i-b*BATCHSIZE;
               VALUETYPE d;
	       d = (pb_X[ind] * pb_X[ind] + pb_Y[ind] * pb_Y[ind]);
	       ENERGY += d ;
	       
               d = STEP / sqrt(d);
	       blasX[i] += pb_X[ind] * d;
               blasY[i] += pb_Y[ind] * d;
	       
               pb_X[ind] = 0;
	       pb_Y[ind] = 0;    
            }
         }
/*
 *    cleanup 
 */
			INDEXTYPE cleanup = (graph.rows/BATCHSIZE) * BATCHSIZE;
			#pragma omp parallel for schedule(dynamic) 
			for(INDEXTYPE i = cleanup; i < graph.rows; i += 1){
                                INDEXTYPE ind = i- cleanup;
                                VALUETYPE fx = 0, fy = 0, distX, distY, dist, dist2;
                                for(INDEXTYPE j = graph.rowptr[i]; j < graph.rowptr[i+1]; j += 1){
                                        int v = graph.colids[j];
                                        distX = blasX[v] - blasX[i];
                                        distY = blasY[v] - blasY[i];
                                        dist = (distX * distX + distY * distY);
                                        dist = sqrt(dist) + 1.0 / dist;
                                        pb_X[ind] += distX * dist;
                                        pb_Y[ind] += distY * dist;
                                }
                                for(INDEXTYPE j = 0; j < i; j += 1){
                                        distX = blasX[j] - blasX[i];
                                        distY = blasY[j] - blasY[i];
                                        dist2 = 1.0 / (distX * distX + distY * distY);
                                        fx += distX * dist2;
                                        fy += distY * dist2;
                                }
                                for(INDEXTYPE j = i+1; j < graph.rows; j += 1){
                                        distX = blasX[j] - blasX[i];
                                        distY = blasY[j] - blasY[i];
                                        dist2 = 1.0 / (distX * distX + distY * distY);
                                        fx += distX * dist2;
                                        fy += distY * dist2;
                                }
                                pb_X[ind] = pb_X[ind] - fx;
                                pb_Y[ind] = pb_Y[ind] - fy;
                        }	
			for(INDEXTYPE i = cleanup; i < graph.rows; i += 1){
                                int ind = i-cleanup;
                                double dist = (pb_X[ind]*pb_X[ind] + pb_Y[ind]*pb_Y[ind]);
                                ENERGY += dist;
				dist = STEP / sqrt(dist);
				blasX[i] += pb_X[ind] * dist;
                                blasY[i] += pb_Y[ind] * dist;
				pb_X[ind] = pb_Y[ind] = 0;
                        }
                        STEP = STEP * 0.999;
                        LOOP++;
                }
                end = omp_get_wtime();
         
         #if NO_FWRITE == 0
                cout << "Efficientunroll Minibatch Size:" << BATCHSIZE  << endl;
                cout << "Efficientunroll Minbatch Energy:" << ENERGY << endl;
                cout << "Efficientunroll Minibatch Parallel Wall time required:" << end - start << endl;
                writeToFileEFF("EFFUR4_"+ to_string(BATCHSIZE)+"PARAOUT" + to_string(LOOP));
         #endif
                result.push_back(ENERGY);
                result.push_back(end - start);
                return result;
   }

#elif defined (MDIM_VEC_UR8)
   
   vector<VALUETYPE> newalgo::EfficientVersionMdim(INDEXTYPE ITERATIONS, 
         INDEXTYPE NUMOFTHREADS, INDEXTYPE BATCHSIZE)
   {
      INDEXTYPE LOOP = 0;
      VALUETYPE start, end, ENERGY, ENERGY0, *pb_X, *pb_Y;
      VALUETYPE STEP = 1.0;
     
      #define UR 8
      
      vector<VALUETYPE> result;
      pb_X = static_cast<VALUETYPE *> (::operator new (sizeof(VALUETYPE[BATCHSIZE])));
      pb_Y = static_cast<VALUETYPE *> (::operator new (sizeof(VALUETYPE[BATCHSIZE])));
      ENERGY0 = ENERGY = numeric_limits<VALUETYPE>::max();

      omp_set_num_threads(NUMOFTHREADS);
/*
 *    time included initDFS 
 */
      start = omp_get_wtime();

      initDFS();
      
      while(LOOP < ITERATIONS)
      {
         ENERGY0 = ENERGY;
         ENERGY = 0;
         for(int i = 0; i < BATCHSIZE; i++)
         {
            pb_X[i] = pb_Y[i] = 0;
         }
	
         
	 for(INDEXTYPE b = 0; b < (graph.rows / BATCHSIZE); b += 1)
         {
	    for(INDEXTYPE i = b * BATCHSIZE; i < (b + 1) * BATCHSIZE; i += UR*VLEN)
            {
	       int ind = i-b*BATCHSIZE;
               register VTYPE vx0, vx1, vx2, vx3, vx4, vx5, vx6, vx7;
               register VTYPE vy0, vy1, vy2, vy3, vy4, vy5, vy6, vy7;
               VTYPE vfx0, vfx1, vfx2, vfx3, vfx4, vfx5, vfx6, vfx7;
               VTYPE vfy0, vfy1, vfy2, vfy3, vfy4, vfy5, vfy6, vfy7;
					
               BCL_vldu(vx0, blasX + i);
	       BCL_vldu(vx1, blasX + i+VLEN);
	       BCL_vldu(vx2, blasX + i+2*VLEN);
	       BCL_vldu(vx3, blasX + i+3*VLEN);
               BCL_vldu(vx4, blasX + i+4*VLEN);
	       BCL_vldu(vx5, blasX + i+5*VLEN);
	       BCL_vldu(vx6, blasX + i+6*VLEN);
	       BCL_vldu(vx7, blasX + i+7*VLEN);

	       BCL_vldu(vy0, blasY + i);
	       BCL_vldu(vy1, blasY + i+1*VLEN);
	       BCL_vldu(vy2, blasY + i+2*VLEN);
	       BCL_vldu(vy3, blasY + i+3*VLEN);
	       BCL_vldu(vy4, blasY + i+4*VLEN);
	       BCL_vldu(vy5, blasY + i+5*VLEN);
	       BCL_vldu(vy6, blasY + i+6*VLEN);
	       BCL_vldu(vy7, blasY + i+7*VLEN);
					
               // init temp sum 
	       BCL_vzero(vfx0); BCL_vzero(vfy0);
	       BCL_vzero(vfx1); BCL_vzero(vfy1);
	       BCL_vzero(vfx2); BCL_vzero(vfy2);
	       BCL_vzero(vfx3); BCL_vzero(vfy3);
	       BCL_vzero(vfx4); BCL_vzero(vfy4);
	       BCL_vzero(vfx5); BCL_vzero(vfy5);
	       BCL_vzero(vfx6); BCL_vzero(vfy6);
	       BCL_vzero(vfx7); BCL_vzero(vfy7);
               
               #pragma omp parallel
               {
                  int id, nthreads; 
                  int chunksize;
                  int j, remainder; 
                  VTYPE vdx0, vdx1, vdx2, vdx3, vdx4, vdx5, vdx6, vdx7;
                  VTYPE vdy0, vdy1, vdy2, vdy3, vdy4, vdy5, vdy6, vdy7;
                  VTYPE vd0, vd1, vd2, vd3, vd4, vd5, vd6, vd7;
/*
 *                NOTE: don't need those temp regs if no_sync used 
 */
                  VTYPE vtfx0, vtfx1, vtfx2, vtfx3, vtfx4, vtfx5, vtfx6, vtfx7;
                  VTYPE vtfy0, vtfy1, vtfy2, vtfy3, vtfy4, vtfy5, vtfy6, vtfy7;

                  const int M = graph.rows; 
                  id = omp_get_thread_num();
                  nthreads = omp_get_num_threads();
                  
                  // init temp sum 
		  BCL_vzero(vtfx0); BCL_vzero(vtfy0);
		  BCL_vzero(vtfx1); BCL_vzero(vtfy1);
		  BCL_vzero(vtfx2); BCL_vzero(vtfy2);
		  BCL_vzero(vtfx3); BCL_vzero(vtfy3);
		  BCL_vzero(vtfx4); BCL_vzero(vtfy4);
		  BCL_vzero(vtfx5); BCL_vzero(vtfy5);
		  BCL_vzero(vtfx6); BCL_vzero(vtfy6);
		  BCL_vzero(vtfx7); BCL_vzero(vtfy7);
/*
 *                j is up to i... lower up case  
 */
                  chunksize = i/nthreads; 
                  
                  for(j = id*chunksize; j < (id+1)*chunksize; j++)
                  {
                     register VTYPE vxj;
                     register VTYPE vyj; 
                     
                     BCL_vbcast(vxj, blasX+j);
                     BCL_vbcast(vyj, blasY+j);
		  
                     //dx0 = xj - x0;
		     BCL_vsub(vdx0, vxj, vx0);
		     BCL_vsub(vdx1, vxj, vx1);
		     BCL_vsub(vdx2, vxj, vx2);
		     BCL_vsub(vdx3, vxj, vx3);
		     BCL_vsub(vdx4, vxj, vx4);
		     BCL_vsub(vdx5, vxj, vx5);
		     BCL_vsub(vdx6, vxj, vx6);
		     BCL_vsub(vdx7, vxj, vx7);
	          
                     // dy0 = yj - y0;
		     BCL_vsub(vdy0, vyj, vy0);
		     BCL_vsub(vdy1, vyj, vy1);
		     BCL_vsub(vdy2, vyj, vy2);
		     BCL_vsub(vdy3, vyj, vy3);
		     BCL_vsub(vdy4, vyj, vy4);
		     BCL_vsub(vdy5, vyj, vy5);
		     BCL_vsub(vdy6, vyj, vy6);
		     BCL_vsub(vdy7, vyj, vy7);
	             
                     //d0 = 1.0 / (dx0 * dx0 + dy0 * dy0);
		     BCL_vmul(vd0, vdx0, vdx0);
		     BCL_vmul(vd1, vdx1, vdx1);
		     BCL_vmul(vd2, vdx2, vdx2);
		     BCL_vmul(vd3, vdx3, vdx3);
		     BCL_vmul(vd4, vdx4, vdx4);
		     BCL_vmul(vd5, vdx5, vdx5);
		     BCL_vmul(vd6, vdx6, vdx6);
		     BCL_vmul(vd7, vdx7, vdx7);
                     
                     BCL_vmac(vd0, vdy0, vdy0);
                     BCL_vmac(vd1, vdy1, vdy1);
                     BCL_vmac(vd2, vdy2, vdy2);
                     BCL_vmac(vd3, vdy3, vdy3);
                     BCL_vmac(vd4, vdy4, vdy4);
                     BCL_vmac(vd5, vdy5, vdy5);
                     BCL_vmac(vd6, vdy6, vdy6);
                     BCL_vmac(vd7, vdy7, vdy7);
                     
                     BCL_vrcp(vd0);
                     BCL_vrcp(vd1);
                     BCL_vrcp(vd2);
                     BCL_vrcp(vd3);
                     BCL_vrcp(vd4);
                     BCL_vrcp(vd5);
                     BCL_vrcp(vd6);
                     BCL_vrcp(vd7);
		     
                     //tfx0 += dx0 * d0;
                     BCL_vmac(vtfx0, vdx0, vd0);
                     BCL_vmac(vtfx1, vdx1, vd1);
                     BCL_vmac(vtfx2, vdx2, vd2);
                     BCL_vmac(vtfx3, vdx3, vd3);
                     BCL_vmac(vtfx4, vdx4, vd4);
                     BCL_vmac(vtfx5, vdx5, vd5);
                     BCL_vmac(vtfx6, vdx6, vd6);
                     BCL_vmac(vtfx7, vdx7, vd7);

                     //tfy0 += dy0 * d0;
                     BCL_vmac(vtfy0, vdy0, vd0);
                     BCL_vmac(vtfy1, vdy1, vd1);
                     BCL_vmac(vtfy2, vdy2, vd2);
                     BCL_vmac(vtfy3, vdy3, vd3);
                     BCL_vmac(vtfy4, vdy4, vd4);
                     BCL_vmac(vtfy5, vdy5, vd5);
                     BCL_vmac(vtfy6, vdy6, vd6);
                     BCL_vmac(vtfy7, vdy7, vd7);
                  }
/*
 *                cleanup of lower  
 */
                  #pragma omp single nowait
                  {
                     for (j=nthreads*chunksize; j < i; j++)
                     {
                        register VTYPE vxj;
                        register VTYPE vyj;
		  
                        BCL_vbcast(vxj, blasX+j);
                        BCL_vbcast(vyj, blasY+j);
                        
                        //dx0 = xj - x0;
		        BCL_vsub(vdx0, vxj, vx0);
		        BCL_vsub(vdx1, vxj, vx1);
		        BCL_vsub(vdx2, vxj, vx2);
		        BCL_vsub(vdx3, vxj, vx3);
		        BCL_vsub(vdx4, vxj, vx4);
		        BCL_vsub(vdx5, vxj, vx5);
		        BCL_vsub(vdx6, vxj, vx6);
		        BCL_vsub(vdx7, vxj, vx7);
	          
                        // dy0 = yj - y0;
		        BCL_vsub(vdy0, vyj, vy0);
		        BCL_vsub(vdy1, vyj, vy1);
		        BCL_vsub(vdy2, vyj, vy2);
		        BCL_vsub(vdy3, vyj, vy3);
		        BCL_vsub(vdy4, vyj, vy4);
		        BCL_vsub(vdy5, vyj, vy5);
		        BCL_vsub(vdy6, vyj, vy6);
		        BCL_vsub(vdy7, vyj, vy7);
	             
                        //d0 = 1.0 / (dx0 * dx0 + dy0 * dy0);
		        BCL_vmul(vd0, vdx0, vdx0);
		        BCL_vmul(vd1, vdx1, vdx1);
		        BCL_vmul(vd2, vdx2, vdx2);
		        BCL_vmul(vd3, vdx3, vdx3);
		        BCL_vmul(vd4, vdx4, vdx4);
		        BCL_vmul(vd5, vdx5, vdx5);
		        BCL_vmul(vd6, vdx6, vdx6);
		        BCL_vmul(vd7, vdx7, vdx7);
                        
                        BCL_vmac(vd0, vdy0, vdy0);
                        BCL_vmac(vd1, vdy1, vdy1);
                        BCL_vmac(vd2, vdy2, vdy2);
                        BCL_vmac(vd3, vdy3, vdy3);
                        BCL_vmac(vd4, vdy4, vdy4);
                        BCL_vmac(vd5, vdy5, vdy5);
                        BCL_vmac(vd6, vdy6, vdy6);
                        BCL_vmac(vd7, vdy7, vdy7);
                        
                        BCL_vrcp(vd0); 
                        BCL_vrcp(vd1); 
                        BCL_vrcp(vd2); 
                        BCL_vrcp(vd3); 
                        BCL_vrcp(vd4); 
                        BCL_vrcp(vd5); 
                        BCL_vrcp(vd6); 
                        BCL_vrcp(vd7); 
		     
                        //tfx0 += dx0 * d0;
                        BCL_vmac(vtfx0, vdx0, vd0);
                        BCL_vmac(vtfx1, vdx1, vd1);
                        BCL_vmac(vtfx2, vdx2, vd2);
                        BCL_vmac(vtfx3, vdx3, vd3);
                        BCL_vmac(vtfx4, vdx4, vd4);
                        BCL_vmac(vtfx5, vdx5, vd5);
                        BCL_vmac(vtfx6, vdx6, vd6);
                        BCL_vmac(vtfx7, vdx7, vd7);

                        //tfy0 += dy0 * d0;
                        BCL_vmac(vtfy0, vdy0, vd0);
                        BCL_vmac(vtfy1, vdy1, vd1);
                        BCL_vmac(vtfy2, vdy2, vd2);
                        BCL_vmac(vtfy3, vdy3, vd3);
                        BCL_vmac(vtfy4, vdy4, vd4);
                        BCL_vmac(vtfy5, vdy5, vd5);
                        BCL_vmac(vtfy6, vdy6, vd6);
                        BCL_vmac(vtfy7, vdy7, vd7);
                     }

/*
 *             location where we need to skip some points i==j
 *             NOTE: its UR iteration, no need to optimize 
 *             Hopefully compiler will unroll and get rid of the conditions
 */
                     for (j = i; j < i+UR*VLEN; j++) 
                     {
                        const unsigned int n = j-i;
                        register VTYPE vxj;
                        register VTYPE vyj;
                        unsigned int ik0, ik1, ik2, ik3, ik4, ik5, ik6, ik7; 
                        const unsigned int AONE = (1<<VLEN)-1; // VLEN=8, AONE=255 
                        
                        BCL_vbcast(vxj, blasX+j);
                        BCL_vbcast(vyj, blasY+j);
                        
/*
 *                      Possible to further optimize.. will think later 
 */
                        ik0 = ( n >= 0 && n < VLEN ) ? 
                           (AONE & (~(1 << (n%VLEN))))  : AONE; 
                        ik1 = ( n >=VLEN && n < VLEN+VLEN )? 
                           (AONE & ~(1 << n%VLEN))  : AONE; 
                        ik2 = ( n >=2*VLEN && n < 2*VLEN+VLEN )? 
                           (AONE & ~(1 << n%VLEN))  : AONE; 
                        ik3 = ( n >=3*VLEN && n < 3*VLEN+VLEN )? 
                           (AONE & ~(1 << n%VLEN))  : AONE; 
                        ik4 = ( n >=4*VLEN && n < 4*VLEN+VLEN )? 
                           (AONE & ~(1 << n%VLEN))  : AONE; 
                        ik5 = ( n >=5*VLEN && n < 5*VLEN+VLEN )? 
                           (AONE & ~(1 << n%VLEN))  : AONE; 
                        ik6 = ( n >=6*VLEN && n < 6*VLEN+VLEN )? 
                           (AONE & ~(1 << n%VLEN))  : AONE; 
                        ik7 = ( n >=7*VLEN && n < 7*VLEN+VLEN )? 
                           (AONE & ~(1 << n%VLEN))  : AONE; 
		        
                        BCL_vsub(vdx0, vxj, vx0);
		        BCL_vsub(vdx1, vxj, vx1);
		        BCL_vsub(vdx2, vxj, vx2);
		        BCL_vsub(vdx3, vxj, vx3);
                        BCL_vsub(vdx4, vxj, vx4);
		        BCL_vsub(vdx5, vxj, vx5);
		        BCL_vsub(vdx6, vxj, vx6);
		        BCL_vsub(vdx7, vxj, vx7);
	          
                        // dy0 = yj - y0;
		        BCL_vsub(vdy0, vyj, vy0);
		        BCL_vsub(vdy1, vyj, vy1);
		        BCL_vsub(vdy2, vyj, vy2);
		        BCL_vsub(vdy3, vyj, vy3);
		        BCL_vsub(vdy4, vyj, vy4);
		        BCL_vsub(vdy5, vyj, vy5);
		        BCL_vsub(vdy6, vyj, vy6);
		        BCL_vsub(vdy7, vyj, vy7);
	             
                        //d0 = 1.0 / (dx0 * dx0 + dy0 * dy0);
		        BCL_vmul(vd0, vdx0, vdx0);
		        BCL_vmul(vd1, vdx1, vdx1);
		        BCL_vmul(vd2, vdx2, vdx2);
		        BCL_vmul(vd3, vdx3, vdx3);
		        BCL_vmul(vd4, vdx4, vdx4);
		        BCL_vmul(vd5, vdx5, vdx5);
		        BCL_vmul(vd6, vdx6, vdx6);
		        BCL_vmul(vd7, vdx7, vdx7);
                        
                        BCL_vmac(vd0, vdy0, vdy0);
                        BCL_vmac(vd1, vdy1, vdy1);
                        BCL_vmac(vd2, vdy2, vdy2);
                        BCL_vmac(vd3, vdy3, vdy3);
                        BCL_vmac(vd4, vdy4, vdy4);
                        BCL_vmac(vd5, vdy5, vdy5);
                        BCL_vmac(vd6, vdy6, vdy6);
                        BCL_vmac(vd7, vdy7, vdy7);
                        
                        BCL_imaskz_vrcp(vd0, ik0); 
                        BCL_imaskz_vrcp(vd1, ik1); 
                        BCL_imaskz_vrcp(vd2, ik2); 
                        BCL_imaskz_vrcp(vd3, ik3); 
                        BCL_imaskz_vrcp(vd4, ik4); 
                        BCL_imaskz_vrcp(vd5, ik5); 
                        BCL_imaskz_vrcp(vd6, ik6); 
                        BCL_imaskz_vrcp(vd7, ik7); 
		     
                        //tfx0 += dx0 * d0;
                        BCL_vmac(vtfx0, vdx0, vd0);
                        BCL_vmac(vtfx1, vdx1, vd1);
                        BCL_vmac(vtfx2, vdx2, vd2);
                        BCL_vmac(vtfx3, vdx3, vd3);
                        BCL_vmac(vtfx4, vdx4, vd4);
                        BCL_vmac(vtfx5, vdx5, vd5);
                        BCL_vmac(vtfx6, vdx6, vd6);
                        BCL_vmac(vtfx7, vdx7, vd7);

                        //tfy0 += dy0 * d0;
                        BCL_vmac(vtfy0, vdy0, vd0);
                        BCL_vmac(vtfy1, vdy1, vd1);
                        BCL_vmac(vtfy2, vdy2, vd2);
                        BCL_vmac(vtfy3, vdy3, vd3);
                        BCL_vmac(vtfy4, vdy4, vd4);
                        BCL_vmac(vtfy5, vdy5, vd5);
                        BCL_vmac(vtfy6, vdy6, vd6);
                        BCL_vmac(vtfy7, vdy7, vd7);
                     }
                  }
/*
 *                Upper part  
 */ 
                  chunksize = (M - i - UR*VLEN)/nthreads; 

                  for(j = (i+UR*VLEN)+id*chunksize; j < (i+UR*VLEN)+(id+1)*chunksize; j++)
                  {
                     register VTYPE vxj;
                     register VTYPE vyj;
                     
                     BCL_vbcast(vxj, blasX+j);
                     BCL_vbcast(vyj, blasY+j);
		  
                     //dx0 = xj - x0;
		     BCL_vsub(vdx0, vxj, vx0);
		     BCL_vsub(vdx1, vxj, vx1);
		     BCL_vsub(vdx2, vxj, vx2);
		     BCL_vsub(vdx3, vxj, vx3);
		     BCL_vsub(vdx4, vxj, vx4);
		     BCL_vsub(vdx5, vxj, vx5);
		     BCL_vsub(vdx6, vxj, vx6);
		     BCL_vsub(vdx7, vxj, vx7);
	          
                     // dy0 = yj - y0;
		     BCL_vsub(vdy0, vyj, vy0);
		     BCL_vsub(vdy1, vyj, vy1);
		     BCL_vsub(vdy2, vyj, vy2);
		     BCL_vsub(vdy3, vyj, vy3);
		     BCL_vsub(vdy4, vyj, vy4);
		     BCL_vsub(vdy5, vyj, vy5);
		     BCL_vsub(vdy6, vyj, vy6);
		     BCL_vsub(vdy7, vyj, vy7);
	             
                     //d0 = 1.0 / (dx0 * dx0 + dy0 * dy0);
		     BCL_vmul(vd0, vdx0, vdx0);
		     BCL_vmul(vd1, vdx1, vdx1);
		     BCL_vmul(vd2, vdx2, vdx2);
		     BCL_vmul(vd3, vdx3, vdx3);
		     BCL_vmul(vd4, vdx4, vdx4);
		     BCL_vmul(vd5, vdx5, vdx5);
		     BCL_vmul(vd6, vdx6, vdx6);
		     BCL_vmul(vd7, vdx7, vdx7);
                     
                     BCL_vmac(vd0, vdy0, vdy0);
                     BCL_vmac(vd1, vdy1, vdy1);
                     BCL_vmac(vd2, vdy2, vdy2);
                     BCL_vmac(vd3, vdy3, vdy3);
                     BCL_vmac(vd4, vdy4, vdy4);
                     BCL_vmac(vd5, vdy5, vdy5);
                     BCL_vmac(vd6, vdy6, vdy6);
                     BCL_vmac(vd7, vdy7, vdy7);
                     
                     BCL_vrcp(vd0); 
                     BCL_vrcp(vd1); 
                     BCL_vrcp(vd2); 
                     BCL_vrcp(vd3); 
                     BCL_vrcp(vd4); 
                     BCL_vrcp(vd5); 
                     BCL_vrcp(vd6); 
                     BCL_vrcp(vd7); 
		     
                     //tfx0 += dx0 * d0;
                     BCL_vmac(vtfx0, vdx0, vd0);
                     BCL_vmac(vtfx1, vdx1, vd1);
                     BCL_vmac(vtfx2, vdx2, vd2);
                     BCL_vmac(vtfx3, vdx3, vd3);
                     BCL_vmac(vtfx4, vdx4, vd4);
                     BCL_vmac(vtfx5, vdx5, vd5);
                     BCL_vmac(vtfx6, vdx6, vd6);
                     BCL_vmac(vtfx7, vdx7, vd7);

                     //tfy0 += dy0 * d0;
                     BCL_vmac(vtfy0, vdy0, vd0);
                     BCL_vmac(vtfy1, vdy1, vd1);
                     BCL_vmac(vtfy2, vdy2, vd2);
                     BCL_vmac(vtfy3, vdy3, vd3);
                     BCL_vmac(vtfy4, vdy4, vd4);
                     BCL_vmac(vtfy5, vdy5, vd5);
                     BCL_vmac(vtfy6, vdy6, vd6);
                     BCL_vmac(vtfy7, vdy7, vd7);
                  }
/*
 *                cleanup 
 */
                  #pragma omp single nowait 
                  {
                     for (j=(i+UR*VLEN)+nthreads*chunksize; j < M; j++)
                     {
                        register VTYPE vxj; 
                        register VTYPE vyj;
		  
                        BCL_vbcast(vxj, blasX+j);
                        BCL_vbcast(vyj, blasY+j);
		  
                        //dx0 = xj - x0;
		        BCL_vsub(vdx0, vxj, vx0);
		        BCL_vsub(vdx1, vxj, vx1);
		        BCL_vsub(vdx2, vxj, vx2);
		        BCL_vsub(vdx3, vxj, vx3);
		        BCL_vsub(vdx4, vxj, vx4);
		        BCL_vsub(vdx5, vxj, vx5);
		        BCL_vsub(vdx6, vxj, vx6);
		        BCL_vsub(vdx7, vxj, vx7);
	          
                        // dy0 = yj - y0;
		        BCL_vsub(vdy0, vyj, vy0);
		        BCL_vsub(vdy1, vyj, vy1);
		        BCL_vsub(vdy2, vyj, vy2);
		        BCL_vsub(vdy3, vyj, vy3);
		        BCL_vsub(vdy4, vyj, vy4);
		        BCL_vsub(vdy5, vyj, vy5);
		        BCL_vsub(vdy6, vyj, vy6);
		        BCL_vsub(vdy7, vyj, vy7);
	             
                        //d0 = 1.0 / (dx0 * dx0 + dy0 * dy0);
		        BCL_vmul(vd0, vdx0, vdx0);
		        BCL_vmul(vd1, vdx1, vdx1);
		        BCL_vmul(vd2, vdx2, vdx2);
		        BCL_vmul(vd3, vdx3, vdx3);
		        BCL_vmul(vd4, vdx4, vdx4);
		        BCL_vmul(vd5, vdx5, vdx5);
		        BCL_vmul(vd6, vdx6, vdx6);
		        BCL_vmul(vd7, vdx7, vdx7);
                        
                        BCL_vmac(vd0, vdy0, vdy0);
                        BCL_vmac(vd1, vdy1, vdy1);
                        BCL_vmac(vd2, vdy2, vdy2);
                        BCL_vmac(vd3, vdy3, vdy3);
                        BCL_vmac(vd4, vdy4, vdy4);
                        BCL_vmac(vd5, vdy5, vdy5);
                        BCL_vmac(vd6, vdy6, vdy6);
                        BCL_vmac(vd7, vdy7, vdy7);
                        
                        BCL_vrcp(vd0); 
                        BCL_vrcp(vd1); 
                        BCL_vrcp(vd2); 
                        BCL_vrcp(vd3); 
                        BCL_vrcp(vd4); 
                        BCL_vrcp(vd5); 
                        BCL_vrcp(vd6); 
                        BCL_vrcp(vd7); 
		     
                        //tfx0 += dx0 * d0;
                        BCL_vmac(vtfx0, vdx0, vd0);
                        BCL_vmac(vtfx1, vdx1, vd1);
                        BCL_vmac(vtfx2, vdx2, vd2);
                        BCL_vmac(vtfx3, vdx3, vd3);
                        BCL_vmac(vtfx4, vdx4, vd4);
                        BCL_vmac(vtfx5, vdx5, vd5);
                        BCL_vmac(vtfx6, vdx6, vd6);
                        BCL_vmac(vtfx7, vdx7, vd7);

                        //tfy0 += dy0 * d0;
                        BCL_vmac(vtfy0, vdy0, vd0);
                        BCL_vmac(vtfy1, vdy1, vd1);
                        BCL_vmac(vtfy2, vdy2, vd2);
                        BCL_vmac(vtfy3, vdy3, vd3);
                        BCL_vmac(vtfy4, vdy4, vd4);
                        BCL_vmac(vtfy5, vdy5, vd5);
                        BCL_vmac(vtfy6, vdy6, vd6);
                        BCL_vmac(vtfy7, vdy7, vd7);
                     }
                  }
/*
 *                Reduction 
 */
                  #pragma omp critical
                  {
                     //fx0 += tfx0; 
                     BCL_vadd(vfx0, vfx0, vtfx0);
                     BCL_vadd(vfx1, vfx1, vtfx1);
                     BCL_vadd(vfx2, vfx2, vtfx2);
                     BCL_vadd(vfx3, vfx3, vtfx3);
                     BCL_vadd(vfx4, vfx4, vtfx4);
                     BCL_vadd(vfx5, vfx5, vtfx5);
                     BCL_vadd(vfx6, vfx6, vtfx6);
                     BCL_vadd(vfx7, vfx7, vtfx7);
                     
                     //fx1 += tfx1; 
                     BCL_vadd(vfy0, vfy0, vtfy0);
                     BCL_vadd(vfy1, vfy1, vtfy1);
                     BCL_vadd(vfy2, vfy2, vtfy2);
                     BCL_vadd(vfy3, vfy3, vtfy3);
                     BCL_vadd(vfy4, vfy4, vtfy4);
                     BCL_vadd(vfy5, vfy5, vtfy5);
                     BCL_vadd(vfy6, vfy6, vtfy6);
                     BCL_vadd(vfy7, vfy7, vtfy7);
                  }
               }
/*
 *             Out of parallel region 
 */
               BCL_vstu(pb_X+ind, vfx0); 
               BCL_vstu(pb_X+ind+VLEN, vfx1); 
               BCL_vstu(pb_X+ind+2*VLEN, vfx2); 
               BCL_vstu(pb_X+ind+3*VLEN, vfx3); 
               BCL_vstu(pb_X+ind+4*VLEN, vfx4); 
               BCL_vstu(pb_X+ind+5*VLEN, vfx5); 
               BCL_vstu(pb_X+ind+6*VLEN, vfx6); 
               BCL_vstu(pb_X+ind+7*VLEN, vfx7); 
               
               BCL_vstu(pb_Y+ind, vfy0); 
               BCL_vstu(pb_Y+ind+VLEN, vfy1); 
               BCL_vstu(pb_Y+ind+2*VLEN, vfy2); 
               BCL_vstu(pb_Y+ind+3*VLEN, vfy3); 
               BCL_vstu(pb_Y+ind+4*VLEN, vfy4); 
               BCL_vstu(pb_Y+ind+5*VLEN, vfy5); 
               BCL_vstu(pb_Y+ind+6*VLEN, vfy6); 
               BCL_vstu(pb_Y+ind+7*VLEN, vfy7); 
            }
            // connected nodes 
	    for(INDEXTYPE i = b * BATCHSIZE; i < (b + 1) * BATCHSIZE; i += 1)
            {   
               int ind = i-b*BATCHSIZE;
               VALUETYPE pbX=0.0, pbY=0.0;
               VALUETYPE dist, distX, distY;
               for(INDEXTYPE j = graph.rowptr[i]; j < graph.rowptr[i+1]; j += 1)
               {
                  int v = graph.colids[j];
                  distX = blasX[v] - blasX[i];
                  distY = blasY[v] - blasY[i];
                  dist = (distX * distX + distY * distY);
                  dist = sqrt(dist) + 1.0 / dist;
                  pbX += distX * dist;
                  pbY += distY * dist;
               }
               pb_X[ind] = pbX - pb_X[ind];
               pb_Y[ind] = pbY - pb_Y[ind];
            }
	    /*printf("After:\n");
            for(INDEXTYPE i = b * BATCHSIZE; i < (b + 1) * BATCHSIZE; i++)
            {
               printf("%d = %lf,", i, pb_X[i-b*BATCHSIZE]);
            }
	    */
	    for(INDEXTYPE i = b * BATCHSIZE; i < (b + 1) * BATCHSIZE; i += 1)
            {
               int ind = i-b*BATCHSIZE;
               VALUETYPE d;
	       d = (pb_X[ind] * pb_X[ind] + pb_Y[ind] * pb_Y[ind]);
	       ENERGY += d ;
	       
               d = STEP / sqrt(d);
	       blasX[i] += pb_X[ind] * d;
               blasY[i] += pb_Y[ind] * d;
	       
               pb_X[ind] = 0;
	       pb_Y[ind] = 0;    
            }
         }
/*
 *    cleanup 
 */
			INDEXTYPE cleanup = (graph.rows/BATCHSIZE) * BATCHSIZE;
			#pragma omp parallel for schedule(dynamic) 
			for(INDEXTYPE i = cleanup; i < graph.rows; i += 1){
                                INDEXTYPE ind = i- cleanup;
                                VALUETYPE fx = 0, fy = 0, distX, distY, dist, dist2;
                                for(INDEXTYPE j = graph.rowptr[i]; j < graph.rowptr[i+1]; j += 1){
                                        int v = graph.colids[j];
                                        distX = blasX[v] - blasX[i];
                                        distY = blasY[v] - blasY[i];
                                        dist = (distX * distX + distY * distY);
                                        dist = sqrt(dist) + 1.0 / dist;
                                        pb_X[ind] += distX * dist;
                                        pb_Y[ind] += distY * dist;
                                }
                                for(INDEXTYPE j = 0; j < i; j += 1){
                                        distX = blasX[j] - blasX[i];
                                        distY = blasY[j] - blasY[i];
                                        dist2 = 1.0 / (distX * distX + distY * distY);
                                        fx += distX * dist2;
                                        fy += distY * dist2;
                                }
                                for(INDEXTYPE j = i+1; j < graph.rows; j += 1){
                                        distX = blasX[j] - blasX[i];
                                        distY = blasY[j] - blasY[i];
                                        dist2 = 1.0 / (distX * distX + distY * distY);
                                        fx += distX * dist2;
                                        fy += distY * dist2;
                                }
                                pb_X[ind] = pb_X[ind] - fx;
                                pb_Y[ind] = pb_Y[ind] - fy;
                        }	
			for(INDEXTYPE i = cleanup; i < graph.rows; i += 1){
                                int ind = i-cleanup;
                                double dist = (pb_X[ind]*pb_X[ind] + pb_Y[ind]*pb_Y[ind]);
                                ENERGY += dist;
				dist = STEP / sqrt(dist);
				blasX[i] += pb_X[ind] * dist;
                                blasY[i] += pb_Y[ind] * dist;
				pb_X[ind] = pb_Y[ind] = 0;
                        }
                        STEP = STEP * 0.999;
                        LOOP++;
                }
                end = omp_get_wtime();
         #if NO_FWRITE == 0
                cout << "Efficientunroll Minibatch Size:" << BATCHSIZE  << endl;
                cout << "Efficientunroll Minbatch Energy:" << ENERGY << endl;
                cout << "Efficientunroll Minibatch Parallel Wall time required:" << end - start << endl;
                writeToFileEFF("EFFUR"+ to_string(BATCHSIZE)+"PARAOUT" + to_string(LOOP));
         #endif
                result.push_back(ENERGY);
                result.push_back(end - start);
                return result;
        }

#elif defined(MDIM_VEC_UR4)
   
   vector<VALUETYPE> newalgo::EfficientVersionMdim(INDEXTYPE ITERATIONS, 
         INDEXTYPE NUMOFTHREADS, INDEXTYPE BATCHSIZE)
   {
      INDEXTYPE LOOP = 0;
      VALUETYPE start, end, ENERGY, ENERGY0, *pb_X, *pb_Y;
      VALUETYPE STEP = 1.0;
      
      
      vector<VALUETYPE> result;
      pb_X = static_cast<VALUETYPE *> (::operator new (sizeof(VALUETYPE[BATCHSIZE])));
      pb_Y = static_cast<VALUETYPE *> (::operator new (sizeof(VALUETYPE[BATCHSIZE])));
      ENERGY0 = ENERGY = numeric_limits<VALUETYPE>::max();

      #define UR 4

      omp_set_num_threads(NUMOFTHREADS);
/*
 *    time included initDFS 
 */
      start = omp_get_wtime();

      initDFS();
      
      while(LOOP < ITERATIONS)
      {
         ENERGY0 = ENERGY;
         ENERGY = 0;
         for(int i = 0; i < BATCHSIZE; i++)
         {
            pb_X[i] = pb_Y[i] = 0;
         }
	
         
	 for(INDEXTYPE b = 0; b < (graph.rows / BATCHSIZE); b += 1)
         {
            // no cleanup since batchsize is multiple of 32 
	    for(INDEXTYPE i = b * BATCHSIZE; i < (b + 1) * BATCHSIZE; i += UR*VLEN)
            {
	       int ind = i-b*BATCHSIZE;
               
               register VTYPE vx0, vx1, vx2, vx3;
               register VTYPE vy0, vy1, vy2, vy3;

               VTYPE vfx0, vfx1, vfx2, vfx3;
               VTYPE vfy0, vfy1, vfy2, vfy3;
					
	       BCL_vldu(vx0, blasX + i);
	       BCL_vldu(vx1, blasX + i+VLEN);
	       BCL_vldu(vx2, blasX + i+2*VLEN);
	       BCL_vldu(vx3, blasX + i+3*VLEN);

	       BCL_vldu(vy0, blasY + i);
	       BCL_vldu(vy1, blasY + i+1*VLEN);
	       BCL_vldu(vy2, blasY + i+2*VLEN);
	       BCL_vldu(vy3, blasY + i+3*VLEN);
					
	       BCL_vzero(vfx0); BCL_vzero(vfy0);
	       BCL_vzero(vfx1); BCL_vzero(vfy1);
	       BCL_vzero(vfx2); BCL_vzero(vfy2);
	       BCL_vzero(vfx3); BCL_vzero(vfy3);
               
               #pragma omp parallel shared(vfx0, vfx1, vfx2, vfx3, vfy0, vfy1, \
                     vfy2, vfy3)
               {
                  int id, nthreads; 
                  int chunksize;
                  int j, remainder; 
                  VTYPE vdx0, vdx1, vdx2, vdx3;
                  VTYPE vdy0, vdy1, vdy2, vdy3;
                  VTYPE vd0, vd1, vd2, vd3;
                  VTYPE vtfx0, vtfx1, vtfx2, vtfx3;
                  VTYPE vtfy0, vtfy1, vtfy2, vtfy3;

                  const int M = graph.rows; 
                  id = omp_get_thread_num();
                  nthreads = omp_get_num_threads();
                  
                  // init temp sum 
		  BCL_vzero(vtfx0); BCL_vzero(vtfy0);
		  BCL_vzero(vtfx1); BCL_vzero(vtfy1);
		  BCL_vzero(vtfx2); BCL_vzero(vtfy2);
		  BCL_vzero(vtfx3); BCL_vzero(vtfy3);
/*
 *                j is up to i... lower up case  
 */
                  chunksize = i/nthreads; 
                  
                  for(j = id*chunksize; j < (id+1)*chunksize; j++)
                  {
                     register VTYPE vxj;
                     register VTYPE vyj; 
                     
                     BCL_vbcast(vxj, blasX+j);
                     BCL_vbcast(vyj, blasY+j);
                     
                     //dx0 = xj - x0;
		     BCL_vsub(vdx0, vxj, vx0);
		     BCL_vsub(vdx1, vxj, vx1);
		     BCL_vsub(vdx2, vxj, vx2);
		     BCL_vsub(vdx3, vxj, vx3);
	          
                     // dy0 = yj - y0;
		     BCL_vsub(vdy0, vyj, vy0);
		     BCL_vsub(vdy1, vyj, vy1);
		     BCL_vsub(vdy2, vyj, vy2);
		     BCL_vsub(vdy3, vyj, vy3);
	             
                     //d0 = 1.0 / (dx0 * dx0 + dy0 * dy0);
		     BCL_vmul(vd0, vdx0, vdx0);
		     BCL_vmul(vd1, vdx1, vdx1);
		     BCL_vmul(vd2, vdx2, vdx2);
		     BCL_vmul(vd3, vdx3, vdx3);
                     
                     BCL_vmac(vd0, vdy0, vdy0);
                     BCL_vmac(vd1, vdy1, vdy1);
                     BCL_vmac(vd2, vdy2, vdy2);
                     BCL_vmac(vd3, vdy3, vdy3);
                     
                     BCL_vrcp(vd0);
                     BCL_vrcp(vd1);
                     BCL_vrcp(vd2);
                     BCL_vrcp(vd3);
                  
                     //tfx0 += dx0 * d0;
                     BCL_vmac(vtfx0, vdx0, vd0);
                     BCL_vmac(vtfx1, vdx1, vd1);
                     BCL_vmac(vtfx2, vdx2, vd2);
                     BCL_vmac(vtfx3, vdx3, vd3);

                     //tfy0 += dy0 * d0;
                     BCL_vmac(vtfy0, vdy0, vd0);
                     BCL_vmac(vtfy1, vdy1, vd1);
                     BCL_vmac(vtfy2, vdy2, vd2);
                     BCL_vmac(vtfy3, vdy3, vd3);
                  }
/*
 *                cleanup of lower  
 */
                  #pragma omp single nowait
                  {
                     for (j=nthreads*chunksize; j < i; j++)
                     {
                        register VTYPE vxj;
                        register VTYPE vyj;
		  
                        BCL_vbcast(vxj, blasX+j);
                        BCL_vbcast(vyj, blasY+j);
		  
                        //dx0 = xj - x0;
		        BCL_vsub(vdx0, vxj, vx0);
		        BCL_vsub(vdx1, vxj, vx1);
		        BCL_vsub(vdx2, vxj, vx2);
		        BCL_vsub(vdx3, vxj, vx3);
	          
                        // dy0 = yj - y0;
		        BCL_vsub(vdy0, vyj, vy0);
		        BCL_vsub(vdy1, vyj, vy1);
		        BCL_vsub(vdy2, vyj, vy2);
		        BCL_vsub(vdy3, vyj, vy3);
	             
                        //d0 = 1.0 / (dx0 * dx0 + dy0 * dy0);
		        BCL_vmul(vd0, vdx0, vdx0);
		        BCL_vmul(vd1, vdx1, vdx1);
		        BCL_vmul(vd2, vdx2, vdx2);
		        BCL_vmul(vd3, vdx3, vdx3);
                        
                        BCL_vmac(vd0, vdy0, vdy0);
                        BCL_vmac(vd1, vdy1, vdy1);
                        BCL_vmac(vd2, vdy2, vdy2);
                        BCL_vmac(vd3, vdy3, vdy3);
                        
                        BCL_vrcp(vd0); 
                        BCL_vrcp(vd1); 
                        BCL_vrcp(vd2); 
                        BCL_vrcp(vd3); 
                  
                        //tfx0 += dx0 * d0;
                        BCL_vmac(vtfx0, vdx0, vd0);
                        BCL_vmac(vtfx1, vdx1, vd1);
                        BCL_vmac(vtfx2, vdx2, vd2);
                        BCL_vmac(vtfx3, vdx3, vd3);

                        //tfy0 += dy0 * d0;
                        BCL_vmac(vtfy0, vdy0, vd0);
                        BCL_vmac(vtfy1, vdy1, vd1);
                        BCL_vmac(vtfy2, vdy2, vd2);
                        BCL_vmac(vtfy3, vdy3, vd3);
                     }

/*
 *             location where we need to skip some points i==j
 *             NOTE: its UR iteration, no need to optimize 
 *             Hopefully compiler will unroll and get rid of the conditions
 */
                     for (j = i; j < i+UR*VLEN; j++) 
                     {
                        const unsigned int n = j-i;
                        register VTYPE vxj;
                        register VTYPE vyj;
                        unsigned int ik0, ik1, ik2, ik3; 
                        const unsigned int AONE = (1<<VLEN)-1; // 255 
                        
                        BCL_vbcast(vxj, blasX+j);
                        BCL_vbcast(vyj, blasY+j);
                        
/*
 *                      Possible to further optimize.. will think later 
 */
                        ik0 = ( n >= 0 && n < VLEN ) ? 
                              (AONE & (~(1 << (n%VLEN)))) : AONE; 
                        ik1 = ( n >= VLEN && n < 2*VLEN )? 
                              (AONE & ~(1 << n%VLEN))     : AONE; 
                        ik2 = ( n >=2*VLEN && n < 3*VLEN )? 
                              (AONE & ~(1 << n%VLEN))     : AONE; 
                        ik3 = ( n >=3*VLEN && n < 4*VLEN )? 
                              (AONE & ~(1 << n%VLEN))     : AONE; 
                        
                        BCL_vsub(vdx0, vxj, vx0);
		        BCL_vsub(vdx1, vxj, vx1);
		        BCL_vsub(vdx2, vxj, vx2);
		        BCL_vsub(vdx3, vxj, vx3);
	          
                        // dy0 = yj - y0;
		        BCL_vsub(vdy0, vyj, vy0);
		        BCL_vsub(vdy1, vyj, vy1);
		        BCL_vsub(vdy2, vyj, vy2);
		        BCL_vsub(vdy3, vyj, vy3);
	             
                        //d0 = 1.0 / (dx0 * dx0 + dy0 * dy0);
		        BCL_vmul(vd0, vdx0, vdx0);
		        BCL_vmul(vd1, vdx1, vdx1);
		        BCL_vmul(vd2, vdx2, vdx2);
		        BCL_vmul(vd3, vdx3, vdx3);
                        
                        BCL_vmac(vd0, vdy0, vdy0);
                        BCL_vmac(vd1, vdy1, vdy1);
                        BCL_vmac(vd2, vdy2, vdy2);
                        BCL_vmac(vd3, vdy3, vdy3);
                        
                        BCL_imaskz_vrcp(vd0, ik0); 
                        BCL_imaskz_vrcp(vd1, ik1); 
                        BCL_imaskz_vrcp(vd2, ik2); 
                        BCL_imaskz_vrcp(vd3, ik3); 
		     
                        //tfx0 += dx0 * d0;
                        BCL_vmac(vtfx0, vdx0, vd0);
                        BCL_vmac(vtfx1, vdx1, vd1);
                        BCL_vmac(vtfx2, vdx2, vd2);
                        BCL_vmac(vtfx3, vdx3, vd3);

                        //tfy0 += dy0 * d0;
                        BCL_vmac(vtfy0, vdy0, vd0);
                        BCL_vmac(vtfy1, vdy1, vd1);
                        BCL_vmac(vtfy2, vdy2, vd2);
                        BCL_vmac(vtfy3, vdy3, vd3);
                     }
                  }
/*
 *                Upper part  
 */ 
                  chunksize = (M - i - UR*VLEN)/nthreads; 

                  for(j = (i+UR*VLEN)+id*chunksize; j < (i+UR*VLEN)+(id+1)*chunksize; j++)
                  {
                     register VTYPE vxj;
                     register VTYPE vyj;
		  
                     BCL_vbcast(vxj, blasX+j);
                     BCL_vbcast(vyj, blasY+j);
                     //dx0 = xj - x0;
		     BCL_vsub(vdx0, vxj, vx0);
		     BCL_vsub(vdx1, vxj, vx1);
		     BCL_vsub(vdx2, vxj, vx2);
		     BCL_vsub(vdx3, vxj, vx3);
	          
                     // dy0 = yj - y0;
		     BCL_vsub(vdy0, vyj, vy0);
		     BCL_vsub(vdy1, vyj, vy1);
		     BCL_vsub(vdy2, vyj, vy2);
		     BCL_vsub(vdy3, vyj, vy3);
	             
                     //d0 = 1.0 / (dx0 * dx0 + dy0 * dy0);
		     BCL_vmul(vd0, vdx0, vdx0);
		     BCL_vmul(vd1, vdx1, vdx1);
		     BCL_vmul(vd2, vdx2, vdx2);
		     BCL_vmul(vd3, vdx3, vdx3);
                     
                     BCL_vmac(vd0, vdy0, vdy0);
                     BCL_vmac(vd1, vdy1, vdy1);
                     BCL_vmac(vd2, vdy2, vdy2);
                     BCL_vmac(vd3, vdy3, vdy3);
                     
                     BCL_vrcp(vd0); 
                     BCL_vrcp(vd1); 
                     BCL_vrcp(vd2); 
                     BCL_vrcp(vd3); 
                  
                     //tfx0 += dx0 * d0;
                     BCL_vmac(vtfx0, vdx0, vd0);
                     BCL_vmac(vtfx1, vdx1, vd1);
                     BCL_vmac(vtfx2, vdx2, vd2);
                     BCL_vmac(vtfx3, vdx3, vd3);

                     //tfy0 += dy0 * d0;
                     BCL_vmac(vtfy0, vdy0, vd0);
                     BCL_vmac(vtfy1, vdy1, vd1);
                     BCL_vmac(vtfy2, vdy2, vd2);
                     BCL_vmac(vtfy3, vdy3, vd3);
                  }
/*
 *                cleanup 
 */
                  #pragma omp single nowait 
                  {
                     for (j=(i+UR*VLEN)+nthreads*chunksize; j < M; j++)
                     {
                        register VTYPE vxj; 
                        register VTYPE vyj;
		  
                        BCL_vbcast(vxj, blasX+j);
                        BCL_vbcast(vyj, blasY+j);
                        
                        //dx0 = xj - x0;
		        BCL_vsub(vdx0, vxj, vx0);
		        BCL_vsub(vdx1, vxj, vx1);
		        BCL_vsub(vdx2, vxj, vx2);
		        BCL_vsub(vdx3, vxj, vx3);
	          
                        // dy0 = yj - y0;
		        BCL_vsub(vdy0, vyj, vy0);
		        BCL_vsub(vdy1, vyj, vy1);
		        BCL_vsub(vdy2, vyj, vy2);
		        BCL_vsub(vdy3, vyj, vy3);
	             
                        //d0 = 1.0 / (dx0 * dx0 + dy0 * dy0);
		        BCL_vmul(vd0, vdx0, vdx0);
		        BCL_vmul(vd1, vdx1, vdx1);
		        BCL_vmul(vd2, vdx2, vdx2);
		        BCL_vmul(vd3, vdx3, vdx3);
                        
                        BCL_vmac(vd0, vdy0, vdy0);
                        BCL_vmac(vd1, vdy1, vdy1);
                        BCL_vmac(vd2, vdy2, vdy2);
                        BCL_vmac(vd3, vdy3, vdy3);
                        
                        BCL_vrcp(vd0); 
                        BCL_vrcp(vd1); 
                        BCL_vrcp(vd2); 
                        BCL_vrcp(vd3); 
		     
                        //tfx0 += dx0 * d0;
                        BCL_vmac(vtfx0, vdx0, vd0);
                        BCL_vmac(vtfx1, vdx1, vd1);
                        BCL_vmac(vtfx2, vdx2, vd2);
                        BCL_vmac(vtfx3, vdx3, vd3);

                        //tfy0 += dy0 * d0;
                        BCL_vmac(vtfy0, vdy0, vd0);
                        BCL_vmac(vtfy1, vdy1, vd1);
                        BCL_vmac(vtfy2, vdy2, vd2);
                        BCL_vmac(vtfy3, vdy3, vd3);
                     }
                  }
/*
 *                Reduction 
 */
                  #pragma omp critical
                  {
                     //fx0 += tfx0; 
                     BCL_vadd(vfx0, vfx0, vtfx0);
                     BCL_vadd(vfx1, vfx1, vtfx1);
                     BCL_vadd(vfx2, vfx2, vtfx2);
                     BCL_vadd(vfx3, vfx3, vtfx3);
		     
                     //fy0 += tfy0; 
                     BCL_vadd(vfy0, vfy0, vtfy0);
                     BCL_vadd(vfy1, vfy1, vtfy1);
                     BCL_vadd(vfy2, vfy2, vtfy2);
                     BCL_vadd(vfy3, vfy3, vtfy3);
                  }
               }
/*
 *             Out of parallel region 
 */
               BCL_vstu(pb_X+ind, vfx0); 
               BCL_vstu(pb_X+ind+VLEN, vfx1); 
               BCL_vstu(pb_X+ind+2*VLEN, vfx2); 
               BCL_vstu(pb_X+ind+3*VLEN, vfx3); 

               BCL_vstu(pb_Y+ind, vfy0); 
               BCL_vstu(pb_Y+ind+VLEN, vfy1); 
               BCL_vstu(pb_Y+ind+2*VLEN, vfy2); 
               BCL_vstu(pb_Y+ind+3*VLEN, vfy3); 
            }
            // connected nodes 
	    for(INDEXTYPE i = b * BATCHSIZE; i < (b + 1) * BATCHSIZE; i += 1)
            {   
               int ind = i-b*BATCHSIZE;
               VALUETYPE pbX=0.0, pbY=0.0;
               VALUETYPE dist, distX, distY;
               for(INDEXTYPE j = graph.rowptr[i]; j < graph.rowptr[i+1]; j += 1)
               {
                  int v = graph.colids[j];
                  distX = blasX[v] - blasX[i];
                  distY = blasY[v] - blasY[i];
                  dist = (distX * distX + distY * distY);
                  dist = sqrt(dist) + 1.0 / dist;
                  pbX += distX * dist;
                  pbY += distY * dist;
               }
               pb_X[ind] = pbX - pb_X[ind];
               pb_Y[ind] = pbY - pb_Y[ind];
            }
	    /*printf("After:\n");
            for(INDEXTYPE i = b * BATCHSIZE; i < (b + 1) * BATCHSIZE; i++)
            {
               printf("%d = %lf,", i, pb_X[i-b*BATCHSIZE]);
            }
	    */
	    for(INDEXTYPE i = b * BATCHSIZE; i < (b + 1) * BATCHSIZE; i += 1)
            {
               int ind = i-b*BATCHSIZE;
               VALUETYPE d;
	       d = (pb_X[ind] * pb_X[ind] + pb_Y[ind] * pb_Y[ind]);
	       ENERGY += d ;
	       
               d = STEP / sqrt(d);
	       blasX[i] += pb_X[ind] * d;
               blasY[i] += pb_Y[ind] * d;
	       
               pb_X[ind] = 0;
	       pb_Y[ind] = 0;    
            }
         }
/*
 *    cleanup 
 */
			INDEXTYPE cleanup = (graph.rows/BATCHSIZE) * BATCHSIZE;
			#pragma omp parallel for schedule(dynamic) 
			for(INDEXTYPE i = cleanup; i < graph.rows; i += 1){
                                INDEXTYPE ind = i- cleanup;
                                VALUETYPE fx = 0, fy = 0, distX, distY, dist, dist2;
                                for(INDEXTYPE j = graph.rowptr[i]; j < graph.rowptr[i+1]; j += 1){
                                        int v = graph.colids[j];
                                        distX = blasX[v] - blasX[i];
                                        distY = blasY[v] - blasY[i];
                                        dist = (distX * distX + distY * distY);
                                        dist = sqrt(dist) + 1.0 / dist;
                                        pb_X[ind] += distX * dist;
                                        pb_Y[ind] += distY * dist;
                                }
                                for(INDEXTYPE j = 0; j < i; j += 1){
                                        distX = blasX[j] - blasX[i];
                                        distY = blasY[j] - blasY[i];
                                        dist2 = 1.0 / (distX * distX + distY * distY);
                                        fx += distX * dist2;
                                        fy += distY * dist2;
                                }
                                for(INDEXTYPE j = i+1; j < graph.rows; j += 1){
                                        distX = blasX[j] - blasX[i];
                                        distY = blasY[j] - blasY[i];
                                        dist2 = 1.0 / (distX * distX + distY * distY);
                                        fx += distX * dist2;
                                        fy += distY * dist2;
                                }
                                pb_X[ind] = pb_X[ind] - fx;
                                pb_Y[ind] = pb_Y[ind] - fy;
                        }	
			for(INDEXTYPE i = cleanup; i < graph.rows; i += 1){
                                int ind = i-cleanup;
                                double dist = (pb_X[ind]*pb_X[ind] + pb_Y[ind]*pb_Y[ind]);
                                ENERGY += dist;
				dist = STEP / sqrt(dist);
				blasX[i] += pb_X[ind] * dist;
                                blasY[i] += pb_Y[ind] * dist;
				pb_X[ind] = pb_Y[ind] = 0;
                        }
                        STEP = STEP * 0.999;
                        LOOP++;
                }
                end = omp_get_wtime();
         #if NO_FWRITE == 0
                cout << "Efficientunroll Minibatch Size:" << BATCHSIZE  << endl;
                cout << "Efficientunroll Minbatch Energy:" << ENERGY << endl;
                cout << "Efficientunroll Minibatch Parallel Wall time required:" << end - start << endl;
                writeToFileEFF("EFFUR4-VEC"+ to_string(BATCHSIZE)+"PARAOUT" + to_string(LOOP));
         #endif
                result.push_back(ENERGY);
                result.push_back(end - start);
                return result;
        }

#elif defined (MDIM_VEC_UR8_NOSYNC)
   
   vector<VALUETYPE> newalgo::EfficientVersionMdim(INDEXTYPE ITERATIONS, 
         INDEXTYPE NUMOFTHREADS, INDEXTYPE BATCHSIZE)
   {
      INDEXTYPE LOOP = 0;
      VALUETYPE start, end, ENERGY, ENERGY0, *pb_X, *pb_Y;
      VALUETYPE STEP = 1.0;
      VALUETYPE *sumX, *sumY;
                  
      vector<VALUETYPE> result;
      pb_X = static_cast<VALUETYPE *> (::operator new (sizeof(VALUETYPE[BATCHSIZE])));
      pb_Y = static_cast<VALUETYPE *> (::operator new (sizeof(VALUETYPE[BATCHSIZE])));
      ENERGY0 = ENERGY = numeric_limits<VALUETYPE>::max();

/*
 *    NOTE: we have different version of reciprocals... I'm defining it here
 *                   NOTE: rcp14_pd inst... precision 2^-14
 *                   rcp28_pd has relative error 2^-28 ... rounding??? 
 *                   NOTE: reciprocal is highly error prone for over/underflow
 *                   FIXME: to compare apple to apple, make div version 
 */
/*
 *    NOTE: to avoid overflow the stack, we allocate memory in heap to 
 *    manage reduction .... 
 *    FIXME: possible to combine it with pb_X and pb_Y???? 
 *           size = NUMOFTHREADS * UNROLL * VLEN
 *    solved the stack overflow by this way 
 *    FIXME: VLEN depends on VALUETYPE... following only works for double as 
 *    a test case 
 */
      #define UR 8
/*
 *    FIXME: allocate 64-byte aligned memory... we can use aligned load/store
 */
      sumX = static_cast<VALUETYPE *> 
               (::operator new (sizeof(VALUETYPE[NUMOFTHREADS*UR*VLEN])));
      sumY = static_cast<VALUETYPE *> 
               (::operator new (sizeof(VALUETYPE[NUMOFTHREADS*UR*VLEN])));

      omp_set_num_threads(NUMOFTHREADS);
/*
 *    time included initDFS 
 */
      start = omp_get_wtime();

      initDFS();
      
      while(LOOP < ITERATIONS)
      {
         ENERGY0 = ENERGY;
         ENERGY = 0;
         for(int i = 0; i < BATCHSIZE; i++)
         {
            pb_X[i] = pb_Y[i] = 0;
         }
	
         
	 for(INDEXTYPE b = 0; b < (graph.rows / BATCHSIZE); b += 1)
         {
	    for(INDEXTYPE i = b * BATCHSIZE; i < (b + 1) * BATCHSIZE; i += UR*VLEN)
            {
	       int ind = i-b*BATCHSIZE;
               int m, n, k;
               register VTYPE vx0, vx1, vx2, vx3, vx4, vx5, vx6, vx7;
               register VTYPE vy0, vy1, vy2, vy3, vy4, vy5, vy6, vy7;
/*
 *             to minimize critical section, we are using extra memory here
 *                Total mem = NUMOFTHREADS * 1KB
 */
               for (m=0; m < NUMOFTHREADS*UR*VLEN; m++)
                  sumX[m] = sumY[m] = 0.0;
	       
               BCL_vldu(vx0, blasX + i);
	       BCL_vldu(vx1, blasX + i+VLEN);
	       BCL_vldu(vx2, blasX + i+2*VLEN);
	       BCL_vldu(vx3, blasX + i+3*VLEN);
               BCL_vldu(vx4, blasX + i+4*VLEN);
	       BCL_vldu(vx5, blasX + i+5*VLEN);
	       BCL_vldu(vx6, blasX + i+6*VLEN);
	       BCL_vldu(vx7, blasX + i+7*VLEN);

	       BCL_vldu(vy0, blasY + i);
	       BCL_vldu(vy1, blasY + i+1*VLEN);
	       BCL_vldu(vy2, blasY + i+2*VLEN);
	       BCL_vldu(vy3, blasY + i+3*VLEN);
	       BCL_vldu(vy4, blasY + i+4*VLEN);
	       BCL_vldu(vy5, blasY + i+5*VLEN);
	       BCL_vldu(vy6, blasY + i+6*VLEN);
	       BCL_vldu(vy7, blasY + i+7*VLEN);

               #pragma omp parallel
               {
                  int id, nthreads; 
                  int chunksize;
                  int j; 
                  VTYPE vdx0, vdx1, vdx2, vdx3, vdx4, vdx5, vdx6, vdx7;
                  VTYPE vdy0, vdy1, vdy2, vdy3, vdy4, vdy5, vdy6, vdy7;
                  VTYPE vd0, vd1, vd2, vd3, vd4, vd5, vd6, vd7;
/*
 *                NOTE: don't need those temp regs if no_sync used 
 */
                  VTYPE vtfx0, vtfx1, vtfx2, vtfx3, vtfx4, vtfx5, vtfx6, vtfx7;
                  VTYPE vtfy0, vtfy1, vtfy2, vtfy3, vtfy4, vtfy5, vtfy6, vtfy7;

                  const int M = graph.rows; 
                  id = omp_get_thread_num();
                  nthreads = omp_get_num_threads();
                  
                  // init temp sum 
		  BCL_vzero(vtfx0); BCL_vzero(vtfy0);
		  BCL_vzero(vtfx1); BCL_vzero(vtfy1);
		  BCL_vzero(vtfx2); BCL_vzero(vtfy2);
		  BCL_vzero(vtfx3); BCL_vzero(vtfy3);
		  BCL_vzero(vtfx4); BCL_vzero(vtfy4);
		  BCL_vzero(vtfx5); BCL_vzero(vtfy5);
		  BCL_vzero(vtfx6); BCL_vzero(vtfy6);
		  BCL_vzero(vtfx7); BCL_vzero(vtfy7);
/*
 *                j is up to i... lower up case  
 */
                  chunksize = i/nthreads; 
                  
                  for(j = id*chunksize; j < (id+1)*chunksize; j++)
                  {
                     register VTYPE vxj;
                     register VTYPE vyj; 
                     
                     BCL_vbcast(vxj, blasX+j);
                     BCL_vbcast(vyj, blasY+j);
		  
                     //dx0 = xj - x0;
		     BCL_vsub(vdx0, vxj, vx0);
		     BCL_vsub(vdx1, vxj, vx1);
		     BCL_vsub(vdx2, vxj, vx2);
		     BCL_vsub(vdx3, vxj, vx3);
		     BCL_vsub(vdx4, vxj, vx4);
		     BCL_vsub(vdx5, vxj, vx5);
		     BCL_vsub(vdx6, vxj, vx6);
		     BCL_vsub(vdx7, vxj, vx7);
	          
                     // dy0 = yj - y0;
		     BCL_vsub(vdy0, vyj, vy0);
		     BCL_vsub(vdy1, vyj, vy1);
		     BCL_vsub(vdy2, vyj, vy2);
		     BCL_vsub(vdy3, vyj, vy3);
		     BCL_vsub(vdy4, vyj, vy4);
		     BCL_vsub(vdy5, vyj, vy5);
		     BCL_vsub(vdy6, vyj, vy6);
		     BCL_vsub(vdy7, vyj, vy7);
	             
                     //d0 = 1.0 / (dx0 * dx0 + dy0 * dy0);
		     BCL_vmul(vd0, vdx0, vdx0);
		     BCL_vmul(vd1, vdx1, vdx1);
		     BCL_vmul(vd2, vdx2, vdx2);
		     BCL_vmul(vd3, vdx3, vdx3);
		     BCL_vmul(vd4, vdx4, vdx4);
		     BCL_vmul(vd5, vdx5, vdx5);
		     BCL_vmul(vd6, vdx6, vdx6);
		     BCL_vmul(vd7, vdx7, vdx7);
                     
                     BCL_vmac(vd0, vdy0, vdy0);
                     BCL_vmac(vd1, vdy1, vdy1);
                     BCL_vmac(vd2, vdy2, vdy2);
                     BCL_vmac(vd3, vdy3, vdy3);
                     BCL_vmac(vd4, vdy4, vdy4);
                     BCL_vmac(vd5, vdy5, vdy5);
                     BCL_vmac(vd6, vdy6, vdy6);
                     BCL_vmac(vd7, vdy7, vdy7);
                     
                     BCL_vrcp(vd0);
                     BCL_vrcp(vd1);
                     BCL_vrcp(vd2);
                     BCL_vrcp(vd3);
                     BCL_vrcp(vd4);
                     BCL_vrcp(vd5);
                     BCL_vrcp(vd6);
                     BCL_vrcp(vd7);
		     
                     //tfx0 += dx0 * d0;
                     BCL_vmac(vtfx0, vdx0, vd0);
                     BCL_vmac(vtfx1, vdx1, vd1);
                     BCL_vmac(vtfx2, vdx2, vd2);
                     BCL_vmac(vtfx3, vdx3, vd3);
                     BCL_vmac(vtfx4, vdx4, vd4);
                     BCL_vmac(vtfx5, vdx5, vd5);
                     BCL_vmac(vtfx6, vdx6, vd6);
                     BCL_vmac(vtfx7, vdx7, vd7);

                     //tfy0 += dy0 * d0;
                     BCL_vmac(vtfy0, vdy0, vd0);
                     BCL_vmac(vtfy1, vdy1, vd1);
                     BCL_vmac(vtfy2, vdy2, vd2);
                     BCL_vmac(vtfy3, vdy3, vd3);
                     BCL_vmac(vtfy4, vdy4, vd4);
                     BCL_vmac(vtfy5, vdy5, vd5);
                     BCL_vmac(vtfy6, vdy6, vd6);
                     BCL_vmac(vtfy7, vdy7, vd7);
                  }
/*
 *                cleanup of lower  
 */
                  #pragma omp single nowait
                  {
                     for (j=nthreads*chunksize; j < i; j++)
                     {
                        register VTYPE vxj;
                        register VTYPE vyj;
		  
                        BCL_vbcast(vxj, blasX+j);
                        BCL_vbcast(vyj, blasY+j);
                        
                        //dx0 = xj - x0;
		        BCL_vsub(vdx0, vxj, vx0);
		        BCL_vsub(vdx1, vxj, vx1);
		        BCL_vsub(vdx2, vxj, vx2);
		        BCL_vsub(vdx3, vxj, vx3);
		        BCL_vsub(vdx4, vxj, vx4);
		        BCL_vsub(vdx5, vxj, vx5);
		        BCL_vsub(vdx6, vxj, vx6);
		        BCL_vsub(vdx7, vxj, vx7);
	          
                        // dy0 = yj - y0;
		        BCL_vsub(vdy0, vyj, vy0);
		        BCL_vsub(vdy1, vyj, vy1);
		        BCL_vsub(vdy2, vyj, vy2);
		        BCL_vsub(vdy3, vyj, vy3);
		        BCL_vsub(vdy4, vyj, vy4);
		        BCL_vsub(vdy5, vyj, vy5);
		        BCL_vsub(vdy6, vyj, vy6);
		        BCL_vsub(vdy7, vyj, vy7);
	             
                        //d0 = 1.0 / (dx0 * dx0 + dy0 * dy0);
		        BCL_vmul(vd0, vdx0, vdx0);
		        BCL_vmul(vd1, vdx1, vdx1);
		        BCL_vmul(vd2, vdx2, vdx2);
		        BCL_vmul(vd3, vdx3, vdx3);
		        BCL_vmul(vd4, vdx4, vdx4);
		        BCL_vmul(vd5, vdx5, vdx5);
		        BCL_vmul(vd6, vdx6, vdx6);
		        BCL_vmul(vd7, vdx7, vdx7);
                        
                        BCL_vmac(vd0, vdy0, vdy0);
                        BCL_vmac(vd1, vdy1, vdy1);
                        BCL_vmac(vd2, vdy2, vdy2);
                        BCL_vmac(vd3, vdy3, vdy3);
                        BCL_vmac(vd4, vdy4, vdy4);
                        BCL_vmac(vd5, vdy5, vdy5);
                        BCL_vmac(vd6, vdy6, vdy6);
                        BCL_vmac(vd7, vdy7, vdy7);
                        
                        BCL_vrcp(vd0); 
                        BCL_vrcp(vd1); 
                        BCL_vrcp(vd2); 
                        BCL_vrcp(vd3); 
                        BCL_vrcp(vd4); 
                        BCL_vrcp(vd5); 
                        BCL_vrcp(vd6); 
                        BCL_vrcp(vd7); 
		     
                        //tfx0 += dx0 * d0;
                        BCL_vmac(vtfx0, vdx0, vd0);
                        BCL_vmac(vtfx1, vdx1, vd1);
                        BCL_vmac(vtfx2, vdx2, vd2);
                        BCL_vmac(vtfx3, vdx3, vd3);
                        BCL_vmac(vtfx4, vdx4, vd4);
                        BCL_vmac(vtfx5, vdx5, vd5);
                        BCL_vmac(vtfx6, vdx6, vd6);
                        BCL_vmac(vtfx7, vdx7, vd7);

                        //tfy0 += dy0 * d0;
                        BCL_vmac(vtfy0, vdy0, vd0);
                        BCL_vmac(vtfy1, vdy1, vd1);
                        BCL_vmac(vtfy2, vdy2, vd2);
                        BCL_vmac(vtfy3, vdy3, vd3);
                        BCL_vmac(vtfy4, vdy4, vd4);
                        BCL_vmac(vtfy5, vdy5, vd5);
                        BCL_vmac(vtfy6, vdy6, vd6);
                        BCL_vmac(vtfy7, vdy7, vd7);
                     }

/*
 *             location where we need to skip some points i==j
 *             NOTE: its UR iteration, no need to optimize 
 *             Hopefully compiler will unroll and get rid of the conditions
 */
                     for (j = i; j < i+UR*VLEN; j++) 
                     {
                        const unsigned int n = j-i;
                        register VTYPE vxj;
                        register VTYPE vyj;
                        unsigned int ik0, ik1, ik2, ik3, ik4, ik5, ik6, ik7; 
                        const unsigned int AONE = (1<<VLEN)-1; // VLEN=8, AONE=255 
                        
                        BCL_vbcast(vxj, blasX+j);
                        BCL_vbcast(vyj, blasY+j);
                        
/*
 *                      Possible to further optimize.. will think later 
 */
                        ik0 = ( n >= 0 && n < VLEN ) ? 
                           (AONE & (~(1 << (n%VLEN))))  : AONE; 
                        ik1 = ( n >=VLEN && n < VLEN+VLEN )? 
                           (AONE & ~(1 << n%VLEN))  : AONE; 
                        ik2 = ( n >=2*VLEN && n < 2*VLEN+VLEN )? 
                           (AONE & ~(1 << n%VLEN))  : AONE; 
                        ik3 = ( n >=3*VLEN && n < 3*VLEN+VLEN )? 
                           (AONE & ~(1 << n%VLEN))  : AONE; 
                        ik4 = ( n >=4*VLEN && n < 4*VLEN+VLEN )? 
                           (AONE & ~(1 << n%VLEN))  : AONE; 
                        ik5 = ( n >=5*VLEN && n < 5*VLEN+VLEN )? 
                           (AONE & ~(1 << n%VLEN))  : AONE; 
                        ik6 = ( n >=6*VLEN && n < 6*VLEN+VLEN )? 
                           (AONE & ~(1 << n%VLEN))  : AONE; 
                        ik7 = ( n >=7*VLEN && n < 7*VLEN+VLEN )? 
                           (AONE & ~(1 << n%VLEN))  : AONE; 
		        
                        BCL_vsub(vdx0, vxj, vx0);
		        BCL_vsub(vdx1, vxj, vx1);
		        BCL_vsub(vdx2, vxj, vx2);
		        BCL_vsub(vdx3, vxj, vx3);
                        BCL_vsub(vdx4, vxj, vx4);
		        BCL_vsub(vdx5, vxj, vx5);
		        BCL_vsub(vdx6, vxj, vx6);
		        BCL_vsub(vdx7, vxj, vx7);
	          
                        // dy0 = yj - y0;
		        BCL_vsub(vdy0, vyj, vy0);
		        BCL_vsub(vdy1, vyj, vy1);
		        BCL_vsub(vdy2, vyj, vy2);
		        BCL_vsub(vdy3, vyj, vy3);
		        BCL_vsub(vdy4, vyj, vy4);
		        BCL_vsub(vdy5, vyj, vy5);
		        BCL_vsub(vdy6, vyj, vy6);
		        BCL_vsub(vdy7, vyj, vy7);
	             
                        //d0 = 1.0 / (dx0 * dx0 + dy0 * dy0);
		        BCL_vmul(vd0, vdx0, vdx0);
		        BCL_vmul(vd1, vdx1, vdx1);
		        BCL_vmul(vd2, vdx2, vdx2);
		        BCL_vmul(vd3, vdx3, vdx3);
		        BCL_vmul(vd4, vdx4, vdx4);
		        BCL_vmul(vd5, vdx5, vdx5);
		        BCL_vmul(vd6, vdx6, vdx6);
		        BCL_vmul(vd7, vdx7, vdx7);
                        
                        BCL_vmac(vd0, vdy0, vdy0);
                        BCL_vmac(vd1, vdy1, vdy1);
                        BCL_vmac(vd2, vdy2, vdy2);
                        BCL_vmac(vd3, vdy3, vdy3);
                        BCL_vmac(vd4, vdy4, vdy4);
                        BCL_vmac(vd5, vdy5, vdy5);
                        BCL_vmac(vd6, vdy6, vdy6);
                        BCL_vmac(vd7, vdy7, vdy7);
                        
                        BCL_imaskz_vrcp(vd0, ik0); 
                        BCL_imaskz_vrcp(vd1, ik1); 
                        BCL_imaskz_vrcp(vd2, ik2); 
                        BCL_imaskz_vrcp(vd3, ik3); 
                        BCL_imaskz_vrcp(vd4, ik4); 
                        BCL_imaskz_vrcp(vd5, ik5); 
                        BCL_imaskz_vrcp(vd6, ik6); 
                        BCL_imaskz_vrcp(vd7, ik7); 
		     
                        //tfx0 += dx0 * d0;
                        BCL_vmac(vtfx0, vdx0, vd0);
                        BCL_vmac(vtfx1, vdx1, vd1);
                        BCL_vmac(vtfx2, vdx2, vd2);
                        BCL_vmac(vtfx3, vdx3, vd3);
                        BCL_vmac(vtfx4, vdx4, vd4);
                        BCL_vmac(vtfx5, vdx5, vd5);
                        BCL_vmac(vtfx6, vdx6, vd6);
                        BCL_vmac(vtfx7, vdx7, vd7);

                        //tfy0 += dy0 * d0;
                        BCL_vmac(vtfy0, vdy0, vd0);
                        BCL_vmac(vtfy1, vdy1, vd1);
                        BCL_vmac(vtfy2, vdy2, vd2);
                        BCL_vmac(vtfy3, vdy3, vd3);
                        BCL_vmac(vtfy4, vdy4, vd4);
                        BCL_vmac(vtfy5, vdy5, vd5);
                        BCL_vmac(vtfy6, vdy6, vd6);
                        BCL_vmac(vtfy7, vdy7, vd7);
                     }
                  }
/*
 *                Upper part  
 *
                  for(INDEXTYPE j = i+8; j < graph.rows; j += 1)
 */ 
                  chunksize = (M - i - UR*VLEN)/nthreads; 

                  for(j = (i+UR*VLEN)+id*chunksize; j < (i+UR*VLEN)+(id+1)*chunksize; j++)
                  {
                     register VTYPE vxj;
                     register VTYPE vyj;
                     
                     BCL_vbcast(vxj, blasX+j);
                     BCL_vbcast(vyj, blasY+j);
		  
                     //dx0 = xj - x0;
		     BCL_vsub(vdx0, vxj, vx0);
		     BCL_vsub(vdx1, vxj, vx1);
		     BCL_vsub(vdx2, vxj, vx2);
		     BCL_vsub(vdx3, vxj, vx3);
		     BCL_vsub(vdx4, vxj, vx4);
		     BCL_vsub(vdx5, vxj, vx5);
		     BCL_vsub(vdx6, vxj, vx6);
		     BCL_vsub(vdx7, vxj, vx7);
	          
                     // dy0 = yj - y0;
		     BCL_vsub(vdy0, vyj, vy0);
		     BCL_vsub(vdy1, vyj, vy1);
		     BCL_vsub(vdy2, vyj, vy2);
		     BCL_vsub(vdy3, vyj, vy3);
		     BCL_vsub(vdy4, vyj, vy4);
		     BCL_vsub(vdy5, vyj, vy5);
		     BCL_vsub(vdy6, vyj, vy6);
		     BCL_vsub(vdy7, vyj, vy7);
	             
                     //d0 = 1.0 / (dx0 * dx0 + dy0 * dy0);
		     BCL_vmul(vd0, vdx0, vdx0);
		     BCL_vmul(vd1, vdx1, vdx1);
		     BCL_vmul(vd2, vdx2, vdx2);
		     BCL_vmul(vd3, vdx3, vdx3);
		     BCL_vmul(vd4, vdx4, vdx4);
		     BCL_vmul(vd5, vdx5, vdx5);
		     BCL_vmul(vd6, vdx6, vdx6);
		     BCL_vmul(vd7, vdx7, vdx7);
                     
                     BCL_vmac(vd0, vdy0, vdy0);
                     BCL_vmac(vd1, vdy1, vdy1);
                     BCL_vmac(vd2, vdy2, vdy2);
                     BCL_vmac(vd3, vdy3, vdy3);
                     BCL_vmac(vd4, vdy4, vdy4);
                     BCL_vmac(vd5, vdy5, vdy5);
                     BCL_vmac(vd6, vdy6, vdy6);
                     BCL_vmac(vd7, vdy7, vdy7);
                     
                     BCL_vrcp(vd0); 
                     BCL_vrcp(vd1); 
                     BCL_vrcp(vd2); 
                     BCL_vrcp(vd3); 
                     BCL_vrcp(vd4); 
                     BCL_vrcp(vd5); 
                     BCL_vrcp(vd6); 
                     BCL_vrcp(vd7); 
		     
                     //tfx0 += dx0 * d0;
                     BCL_vmac(vtfx0, vdx0, vd0);
                     BCL_vmac(vtfx1, vdx1, vd1);
                     BCL_vmac(vtfx2, vdx2, vd2);
                     BCL_vmac(vtfx3, vdx3, vd3);
                     BCL_vmac(vtfx4, vdx4, vd4);
                     BCL_vmac(vtfx5, vdx5, vd5);
                     BCL_vmac(vtfx6, vdx6, vd6);
                     BCL_vmac(vtfx7, vdx7, vd7);

                     //tfy0 += dy0 * d0;
                     BCL_vmac(vtfy0, vdy0, vd0);
                     BCL_vmac(vtfy1, vdy1, vd1);
                     BCL_vmac(vtfy2, vdy2, vd2);
                     BCL_vmac(vtfy3, vdy3, vd3);
                     BCL_vmac(vtfy4, vdy4, vd4);
                     BCL_vmac(vtfy5, vdy5, vd5);
                     BCL_vmac(vtfy6, vdy6, vd6);
                     BCL_vmac(vtfy7, vdy7, vd7);
                  }
/*
 *                cleanup 
 */
                  #pragma omp single nowait 
                  {
                     for (j=(i+UR*VLEN)+nthreads*chunksize; j < M; j++)
                     {
                        register VTYPE vxj; 
                        register VTYPE vyj;
		  
                        BCL_vbcast(vxj, blasX+j);
                        BCL_vbcast(vyj, blasY+j);
		  
                        //dx0 = xj - x0;
		        BCL_vsub(vdx0, vxj, vx0);
		        BCL_vsub(vdx1, vxj, vx1);
		        BCL_vsub(vdx2, vxj, vx2);
		        BCL_vsub(vdx3, vxj, vx3);
		        BCL_vsub(vdx4, vxj, vx4);
		        BCL_vsub(vdx5, vxj, vx5);
		        BCL_vsub(vdx6, vxj, vx6);
		        BCL_vsub(vdx7, vxj, vx7);
	          
                        // dy0 = yj - y0;
		        BCL_vsub(vdy0, vyj, vy0);
		        BCL_vsub(vdy1, vyj, vy1);
		        BCL_vsub(vdy2, vyj, vy2);
		        BCL_vsub(vdy3, vyj, vy3);
		        BCL_vsub(vdy4, vyj, vy4);
		        BCL_vsub(vdy5, vyj, vy5);
		        BCL_vsub(vdy6, vyj, vy6);
		        BCL_vsub(vdy7, vyj, vy7);
	             
                        //d0 = 1.0 / (dx0 * dx0 + dy0 * dy0);
		        BCL_vmul(vd0, vdx0, vdx0);
		        BCL_vmul(vd1, vdx1, vdx1);
		        BCL_vmul(vd2, vdx2, vdx2);
		        BCL_vmul(vd3, vdx3, vdx3);
		        BCL_vmul(vd4, vdx4, vdx4);
		        BCL_vmul(vd5, vdx5, vdx5);
		        BCL_vmul(vd6, vdx6, vdx6);
		        BCL_vmul(vd7, vdx7, vdx7);
                        
                        BCL_vmac(vd0, vdy0, vdy0);
                        BCL_vmac(vd1, vdy1, vdy1);
                        BCL_vmac(vd2, vdy2, vdy2);
                        BCL_vmac(vd3, vdy3, vdy3);
                        BCL_vmac(vd4, vdy4, vdy4);
                        BCL_vmac(vd5, vdy5, vdy5);
                        BCL_vmac(vd6, vdy6, vdy6);
                        BCL_vmac(vd7, vdy7, vdy7);
                        
                        BCL_vrcp(vd0); 
                        BCL_vrcp(vd1); 
                        BCL_vrcp(vd2); 
                        BCL_vrcp(vd3); 
                        BCL_vrcp(vd4); 
                        BCL_vrcp(vd5); 
                        BCL_vrcp(vd6); 
                        BCL_vrcp(vd7); 
		     
                        //tfx0 += dx0 * d0;
                        BCL_vmac(vtfx0, vdx0, vd0);
                        BCL_vmac(vtfx1, vdx1, vd1);
                        BCL_vmac(vtfx2, vdx2, vd2);
                        BCL_vmac(vtfx3, vdx3, vd3);
                        BCL_vmac(vtfx4, vdx4, vd4);
                        BCL_vmac(vtfx5, vdx5, vd5);
                        BCL_vmac(vtfx6, vdx6, vd6);
                        BCL_vmac(vtfx7, vdx7, vd7);

                        //tfy0 += dy0 * d0;
                        BCL_vmac(vtfy0, vdy0, vd0);
                        BCL_vmac(vtfy1, vdy1, vd1);
                        BCL_vmac(vtfy2, vdy2, vd2);
                        BCL_vmac(vtfy3, vdy3, vd3);
                        BCL_vmac(vtfy4, vdy4, vd4);
                        BCL_vmac(vtfy5, vdy5, vd5);
                        BCL_vmac(vtfy6, vdy6, vd6);
                        BCL_vmac(vtfy7, vdy7, vd7);
                     }
                  }
/*
 *                Reduction without synchronization  
 */
                  {
                     register VTYPE tv; 
                     double *ptr = sumX + id*UR*VLEN;
                     
                     //fx0 += tfx0; 
                     BCL_vldu(tv, ptr);
                     BCL_vadd(vtfx0, vtfx0, tv);
                     BCL_vstu(ptr, vtfx0);
                     
                     BCL_vldu(tv, ptr+1*VLEN);
                     BCL_vadd(vtfx1, vtfx1, tv);
                     BCL_vstu(ptr+1*VLEN, vtfx1);
                     
                     BCL_vldu(tv, ptr+2*VLEN);
                     BCL_vadd(vtfx2, vtfx2, tv);
                     BCL_vstu(ptr+2*VLEN, vtfx2);

                     BCL_vldu(tv, ptr+3*VLEN);
                     BCL_vadd(vtfx3, vtfx3, tv);
                     BCL_vstu(ptr+3*VLEN, vtfx3);
                     
                     BCL_vldu(tv, ptr+4*VLEN);
                     BCL_vadd(vtfx4, vtfx4, tv);
                     BCL_vstu(ptr+4*VLEN, vtfx4);
                     
                     BCL_vldu(tv, ptr+5*VLEN);
                     BCL_vadd(vtfx5, vtfx5, tv);
                     BCL_vstu(ptr+5*VLEN, vtfx5);
                     
                     BCL_vldu(tv, ptr+6*VLEN);
                     BCL_vadd(vtfx6, vtfx6, tv);
                     BCL_vstu(ptr+6*VLEN, vtfx6);

                     BCL_vldu(tv, ptr+7*VLEN);
                     BCL_vadd(vtfx7, vtfx7, tv);
                     BCL_vstu(ptr+7*VLEN, vtfx7);
                     
                     //fy1 += tfy1; 
                     ptr = (double*) sumY + id*UR*VLEN;
                     BCL_vldu(tv, ptr);
                     BCL_vadd(vtfy0, vtfy0, tv);
                     BCL_vstu(ptr, vtfy0);
                     
                     BCL_vldu(tv, ptr+1*VLEN);
                     BCL_vadd(vtfy1, vtfy1, tv);
                     BCL_vstu(ptr+1*VLEN, vtfy1);
                     
                     BCL_vldu(tv, ptr+2*VLEN);
                     BCL_vadd(vtfy2, vtfy2, tv);
                     BCL_vstu(ptr+2*VLEN, vtfy2);

                     BCL_vldu(tv, ptr+3*VLEN);
                     BCL_vadd(vtfy3, vtfy3, tv);
                     BCL_vstu(ptr+3*VLEN, vtfy3);
                     
                     BCL_vldu(tv, ptr+4*VLEN);
                     BCL_vadd(vtfy4, vtfy4, tv);
                     BCL_vstu(ptr+4*VLEN, vtfy4);
                     
                     BCL_vldu(tv, ptr+5*VLEN);
                     BCL_vadd(vtfy5, vtfy5, tv);
                     BCL_vstu(ptr+5*VLEN, vtfy5);
                     
                     BCL_vldu(tv, ptr+6*VLEN);
                     BCL_vadd(vtfy6, vtfy6, tv);
                     BCL_vstu(ptr+6*VLEN, vtfy6);

                     BCL_vldu(tv, ptr+7*VLEN);
                     BCL_vadd(vtfy7, vtfy7, tv);
                     BCL_vstu(ptr+7*VLEN, vtfy7);
                  }
               }
/*
 *             Out of parallel region 
 */
/*
 *             sum the results up: 
 *             FIXME: use intrinsic to speed it up 
 */
               for(m=1; m < NUMOFTHREADS; m++)
               {
                  for (n=0; n < UR*VLEN; n++)
                  {
                     sumX[n] += sumX[m*UR*VLEN+n];
                     sumY[n] += sumY[m*UR*VLEN+n];
                  }
               }

               double *ptr = &sumX[0]; 
               VTYPE tvsum;
               
               BCL_vldu(tvsum, ptr);
               BCL_vstu(pb_X+ind, tvsum); 
               
               BCL_vldu(tvsum, ptr+1*VLEN);
               BCL_vstu(pb_X+ind+VLEN, tvsum); 
               
               BCL_vldu(tvsum, ptr+2*VLEN);
               BCL_vstu(pb_X+ind+2*VLEN, tvsum);

               BCL_vldu(tvsum, ptr+3*VLEN);
               BCL_vstu(pb_X+ind+3*VLEN, tvsum); 
               
               BCL_vldu(tvsum, ptr+4*VLEN);
               BCL_vstu(pb_X+ind+4*VLEN, tvsum); 
               
               BCL_vldu(tvsum, ptr+5*VLEN);
               BCL_vstu(pb_X+ind+5*VLEN, tvsum); 
               
               BCL_vldu(tvsum, ptr+6*VLEN);
               BCL_vstu(pb_X+ind+6*VLEN, tvsum); 
               
               BCL_vldu(tvsum, ptr+7*VLEN);
               BCL_vstu(pb_X+ind+7*VLEN, tvsum); 
               
               // Y  
               ptr = &sumY[0]; 
               BCL_vldu(tvsum, ptr);
               BCL_vstu(pb_Y+ind, tvsum); 
               
               BCL_vldu(tvsum, ptr+VLEN);
               BCL_vstu(pb_Y+ind+VLEN, tvsum); 
               
               BCL_vldu(tvsum, ptr+2*VLEN);
               BCL_vstu(pb_Y+ind+2*VLEN, tvsum);

               BCL_vldu(tvsum, ptr+3*VLEN);
               BCL_vstu(pb_Y+ind+3*VLEN, tvsum); 
               
               BCL_vldu(tvsum, ptr+4*VLEN);
               BCL_vstu(pb_Y+ind+4*VLEN, tvsum); 
               
               BCL_vldu(tvsum, ptr+5*VLEN);
               BCL_vstu(pb_Y+ind+5*VLEN, tvsum); 
               
               BCL_vldu(tvsum, ptr+6*VLEN);
               BCL_vstu(pb_Y+ind+6*VLEN, tvsum); 
               
               BCL_vldu(tvsum, ptr+7*VLEN);
               BCL_vstu(pb_Y+ind+7*VLEN, tvsum); 
               
            }
            // connected nodes 
	    for(INDEXTYPE i = b * BATCHSIZE; i < (b + 1) * BATCHSIZE; i += 1)
            {   
               int ind = i-b*BATCHSIZE;
               VALUETYPE pbX=0.0, pbY=0.0;
               VALUETYPE dist, distX, distY;
               for(INDEXTYPE j = graph.rowptr[i]; j < graph.rowptr[i+1]; j += 1)
               {
                  int v = graph.colids[j];
                  distX = blasX[v] - blasX[i];
                  distY = blasY[v] - blasY[i];
                  dist = (distX * distX + distY * distY);
                  dist = sqrt(dist) + 1.0 / dist;
                  pbX += distX * dist;
                  pbY += distY * dist;
               }
               pb_X[ind] = pbX - pb_X[ind];
               pb_Y[ind] = pbY - pb_Y[ind];
            }
	    /*printf("After:\n");
            for(INDEXTYPE i = b * BATCHSIZE; i < (b + 1) * BATCHSIZE; i++)
            {
               printf("%d = %lf,", i, pb_X[i-b*BATCHSIZE]);
            }
	    */
	    for(INDEXTYPE i = b * BATCHSIZE; i < (b + 1) * BATCHSIZE; i += 1)
            {
               int ind = i-b*BATCHSIZE;
               VALUETYPE d;
	       d = (pb_X[ind] * pb_X[ind] + pb_Y[ind] * pb_Y[ind]);
	       ENERGY += d ;
	       
               d = STEP / sqrt(d);
	       blasX[i] += pb_X[ind] * d;
               blasY[i] += pb_Y[ind] * d;
	       
               pb_X[ind] = 0;
	       pb_Y[ind] = 0;    
            }
         }
/*
 *    cleanup 
 */
			INDEXTYPE cleanup = (graph.rows/BATCHSIZE) * BATCHSIZE;
			#pragma omp parallel for schedule(dynamic) 
			for(INDEXTYPE i = cleanup; i < graph.rows; i += 1){
                                INDEXTYPE ind = i- cleanup;
                                VALUETYPE fx = 0, fy = 0, distX, distY, dist, dist2;
                                for(INDEXTYPE j = graph.rowptr[i]; j < graph.rowptr[i+1]; j += 1){
                                        int v = graph.colids[j];
                                        distX = blasX[v] - blasX[i];
                                        distY = blasY[v] - blasY[i];
                                        dist = (distX * distX + distY * distY);
                                        dist = sqrt(dist) + 1.0 / dist;
                                        pb_X[ind] += distX * dist;
                                        pb_Y[ind] += distY * dist;
                                }
                                for(INDEXTYPE j = 0; j < i; j += 1){
                                        distX = blasX[j] - blasX[i];
                                        distY = blasY[j] - blasY[i];
                                        dist2 = 1.0 / (distX * distX + distY * distY);
                                        fx += distX * dist2;
                                        fy += distY * dist2;
                                }
                                for(INDEXTYPE j = i+1; j < graph.rows; j += 1){
                                        distX = blasX[j] - blasX[i];
                                        distY = blasY[j] - blasY[i];
                                        dist2 = 1.0 / (distX * distX + distY * distY);
                                        fx += distX * dist2;
                                        fy += distY * dist2;
                                }
                                pb_X[ind] = pb_X[ind] - fx;
                                pb_Y[ind] = pb_Y[ind] - fy;
                        }	
			for(INDEXTYPE i = cleanup; i < graph.rows; i += 1){
                                int ind = i-cleanup;
                                double dist = (pb_X[ind]*pb_X[ind] + pb_Y[ind]*pb_Y[ind]);
                                ENERGY += dist;
				dist = STEP / sqrt(dist);
				blasX[i] += pb_X[ind] * dist;
                                blasY[i] += pb_Y[ind] * dist;
				pb_X[ind] = pb_Y[ind] = 0;
                        }
                        STEP = STEP * 0.999;
                        LOOP++;
                }
                end = omp_get_wtime();
         
         #if NO_FWRITE == 0
                cout << "Efficientunroll Minibatch Size:" << BATCHSIZE  << endl;
                cout << "Efficientunroll Minbatch Energy:" << ENERGY << endl;
                cout << "Efficientunroll Minibatch Parallel Wall time required:" << end - start << endl;
                writeToFileEFF("EFFUR8_VEC_NOSYNC"+ to_string(BATCHSIZE)+"PARAOUT" + to_string(LOOP));
         #endif
                result.push_back(ENERGY);
                result.push_back(end - start);
                return result;
        }

#elif defined(MDIM_VEC_UR4_NOSYNC)

   vector<VALUETYPE> newalgo::EfficientVersionMdim(INDEXTYPE ITERATIONS, 
         INDEXTYPE NUMOFTHREADS, INDEXTYPE BATCHSIZE)
   {
      INDEXTYPE LOOP = 0;
      VALUETYPE start, end, ENERGY, ENERGY0, *pb_X, *pb_Y;
      VALUETYPE STEP = 1.0;
      VALUETYPE *sumX, *sumY;
      
      
      vector<VALUETYPE> result;
      pb_X = static_cast<VALUETYPE *> (::operator new (sizeof(VALUETYPE[BATCHSIZE])));
      pb_Y = static_cast<VALUETYPE *> (::operator new (sizeof(VALUETYPE[BATCHSIZE])));
      ENERGY0 = ENERGY = numeric_limits<VALUETYPE>::max();

/*
 *    NOTE: we have different version of reciprocals... I'm defining it here
 *                   NOTE: rcp14_pd inst... precision 2^-14
 *                   rcp28_pd has relative error 2^-28 ... rounding??? 
 *                   NOTE: reciprocal is highly error prone for over/underflow
 */
      #define UR 4
/*
 *    NOTE: to avoid overflow the stack, we allocate memory in heap to 
 *    manage reduction .... 
 *    FIXME: possible to combine it with pb_X and pb_Y???? 
 *           size = NUMOFTHREADS * UNROLL * VLEN
 *    solved the stack overflow by this way
 *    NEED C++17 to use aligned new 
 */
      sumX = static_cast<VALUETYPE *> 
               (::operator new (sizeof(VALUETYPE[NUMOFTHREADS*UR*VLEN])));
      sumY = static_cast<VALUETYPE *> 
               (::operator new (sizeof(VALUETYPE[NUMOFTHREADS*UR*VLEN])));

      omp_set_num_threads(NUMOFTHREADS);
/*
 *    time included initDFS 
 */
      start = omp_get_wtime();

      initDFS();
      
      while(LOOP < ITERATIONS)
      {
         ENERGY0 = ENERGY;
         ENERGY = 0;
         for(int i = 0; i < BATCHSIZE; i++)
         {
            pb_X[i] = pb_Y[i] = 0;
         }
         
	 for(INDEXTYPE b = 0; b < (graph.rows / BATCHSIZE); b += 1)
         {
	    for(INDEXTYPE i = b * BATCHSIZE; i < (b + 1) * BATCHSIZE; i += UR*VLEN)
            {
	       int ind = i-b*BATCHSIZE;
               int m, n, k;
               register VTYPE vx0, vx1, vx2, vx3;
               register VTYPE vy0, vy1, vy2, vy3;
/*
 *             to minimize critical section, we are using extra memory here
 *             FIXME: use intrinsic to optimize it, like: XoR  
 */
               for (m=0; m < NUMOFTHREADS*UR*VLEN; m++)
                  sumX[m] = sumY[m] = 0.0;

	       BCL_vldu(vx0, blasX + i);
	       BCL_vldu(vx1, blasX + i+VLEN);
	       BCL_vldu(vx2, blasX + i+2*VLEN);
	       BCL_vldu(vx3, blasX + i+3*VLEN);

	       BCL_vldu(vy0, blasY + i);
	       BCL_vldu(vy1, blasY + i+1*VLEN);
	       BCL_vldu(vy2, blasY + i+2*VLEN);
	       BCL_vldu(vy3, blasY + i+3*VLEN);
/*
 *             starting of parallel region 
 */
               #pragma omp parallel shared(sumX, sumY)
               {
                  int id, nthreads; 
                  int chunksize;
                  int j; 
                  
                  VTYPE vdx0, vdx1, vdx2, vdx3;
                  VTYPE vdy0, vdy1, vdy2, vdy3;
                  VTYPE vd0, vd1, vd2, vd3;
                  VTYPE vtfx0, vtfx1, vtfx2, vtfx3;
                  VTYPE vtfy0, vtfy1, vtfy2, vtfy3;

                  const int M = graph.rows; 
                  id = omp_get_thread_num();
                  nthreads = omp_get_num_threads();
                  
                  // init temp sum 
		  BCL_vzero(vtfx0); BCL_vzero(vtfy0);
		  BCL_vzero(vtfx1); BCL_vzero(vtfy1);
		  BCL_vzero(vtfx2); BCL_vzero(vtfy2);
		  BCL_vzero(vtfx3); BCL_vzero(vtfy3);
/*
 *                j is up to i... lower up case  
 */
                  chunksize = i/nthreads; 
                  
                  for(j = id*chunksize; j < (id+1)*chunksize; j++)
                  {
                     register VTYPE vxj;
                     register VTYPE vyj; 
                     
                     BCL_vbcast(vxj, blasX+j);
                     BCL_vbcast(vyj, blasY+j);
		  
                     //dx0 = xj - x0;
		     BCL_vsub(vdx0, vxj, vx0);
		     BCL_vsub(vdx1, vxj, vx1);
		     BCL_vsub(vdx2, vxj, vx2);
		     BCL_vsub(vdx3, vxj, vx3);
	          
                     // dy0 = yj - y0;
		     BCL_vsub(vdy0, vyj, vy0);
		     BCL_vsub(vdy1, vyj, vy1);
		     BCL_vsub(vdy2, vyj, vy2);
		     BCL_vsub(vdy3, vyj, vy3);
	             
                     //d0 = 1.0 / (dx0 * dx0 + dy0 * dy0);
		     BCL_vmul(vd0, vdx0, vdx0);
		     BCL_vmul(vd1, vdx1, vdx1);
		     BCL_vmul(vd2, vdx2, vdx2);
		     BCL_vmul(vd3, vdx3, vdx3);
                     
                     BCL_vmac(vd0, vdy0, vdy0);
                     BCL_vmac(vd1, vdy1, vdy1);
                     BCL_vmac(vd2, vdy2, vdy2);
                     BCL_vmac(vd3, vdy3, vdy3);
                     
                     BCL_vrcp(vd0);
                     BCL_vrcp(vd1);
                     BCL_vrcp(vd2);
                     BCL_vrcp(vd3);
                  
                     //tfx0 += dx0 * d0;
                     BCL_vmac(vtfx0, vdx0, vd0);
                     BCL_vmac(vtfx1, vdx1, vd1);
                     BCL_vmac(vtfx2, vdx2, vd2);
                     BCL_vmac(vtfx3, vdx3, vd3);

                     //tfy0 += dy0 * d0;
                     BCL_vmac(vtfy0, vdy0, vd0);
                     BCL_vmac(vtfy1, vdy1, vd1);
                     BCL_vmac(vtfy2, vdy2, vd2);
                     BCL_vmac(vtfy3, vdy3, vd3);
                  }
/*
 *                cleanup of lower  
 */
                  #pragma omp single nowait
                  {
                     for (j=nthreads*chunksize; j < i; j++)
                     {
                        register VTYPE vxj;
                        register VTYPE vyj;
		  
                        BCL_vbcast(vxj, blasX+j);
                        BCL_vbcast(vyj, blasY+j);
                    
                        //dx0 = xj - x0;
		        BCL_vsub(vdx0, vxj, vx0);
		        BCL_vsub(vdx1, vxj, vx1);
		        BCL_vsub(vdx2, vxj, vx2);
		        BCL_vsub(vdx3, vxj, vx3);
	          
                        // dy0 = yj - y0;
		        BCL_vsub(vdy0, vyj, vy0);
		        BCL_vsub(vdy1, vyj, vy1);
		        BCL_vsub(vdy2, vyj, vy2);
		        BCL_vsub(vdy3, vyj, vy3);
	             
                        //d0 = 1.0 / (dx0 * dx0 + dy0 * dy0);
		        BCL_vmul(vd0, vdx0, vdx0);
		        BCL_vmul(vd1, vdx1, vdx1);
		        BCL_vmul(vd2, vdx2, vdx2);
		        BCL_vmul(vd3, vdx3, vdx3);
                        
                        BCL_vmac(vd0, vdy0, vdy0);
                        BCL_vmac(vd1, vdy1, vdy1);
                        BCL_vmac(vd2, vdy2, vdy2);
                        BCL_vmac(vd3, vdy3, vdy3);
                        
                        BCL_vrcp(vd0); 
                        BCL_vrcp(vd1); 
                        BCL_vrcp(vd2); 
                        BCL_vrcp(vd3); 
                  
                        //tfx0 += dx0 * d0;
                        BCL_vmac(vtfx0, vdx0, vd0);
                        BCL_vmac(vtfx1, vdx1, vd1);
                        BCL_vmac(vtfx2, vdx2, vd2);
                        BCL_vmac(vtfx3, vdx3, vd3);

                        //tfy0 += dy0 * d0;
                        BCL_vmac(vtfy0, vdy0, vd0);
                        BCL_vmac(vtfy1, vdy1, vd1);
                        BCL_vmac(vtfy2, vdy2, vd2);
                        BCL_vmac(vtfy3, vdy3, vd3);
                     }

/*
 *             location where we need to skip some points i==j
 *             NOTE: its UR iteration, no need to optimize 
 */
                     for (j = i; j < i+UR*VLEN; j++) 
                     {
                        const unsigned int n = j-i;
                        register VTYPE vxj;
                        register VTYPE vyj;
                        unsigned int ik0, ik1, ik2, ik3; 
                        const unsigned int AONE = (1<<VLEN)-1; // 255 
                        
                        BCL_vbcast(vxj, blasX+j);
                        BCL_vbcast(vyj, blasY+j);
/*
 *                      Possible to further optimize.. will think later 
 */
                     #if 0
                        ik0 = ( n >= 0 && n < 8 ) ? (255 & (~(1 << (n%8))))  : 255; 
                        ik1 = ( n >=8 && n < 8+8 )? (255 & ~(1 << n%8))  : 255; 
                        ik2 = ( n >=2*8 && n < 2*8+8 )? (255 & ~(1 << n%8))  : 255; 
                        ik3 = ( n >=3*8 && n < 3*8+8 )? (255 & ~(1 << n%8))  : 255; 
                     #endif
                        ik0 = ( n >= 0 && n < VLEN ) ? 
                              (AONE & (~(1 << (n%VLEN)))) : AONE; 
                        ik1 = ( n >= VLEN && n < 2*VLEN )? 
                              (AONE & ~(1 << n%VLEN))     : AONE; 
                        ik2 = ( n >=2*VLEN && n < 3*VLEN )? 
                              (AONE & ~(1 << n%VLEN))     : AONE; 
                        ik3 = ( n >=3*VLEN && n < 4*VLEN )? 
                              (AONE & ~(1 << n%VLEN))     : AONE; 
                        
		        
                        BCL_vsub(vdx0, vxj, vx0);
		        BCL_vsub(vdx1, vxj, vx1);
		        BCL_vsub(vdx2, vxj, vx2);
		        BCL_vsub(vdx3, vxj, vx3);
	          
                        // dy0 = yj - y0;
		        BCL_vsub(vdy0, vyj, vy0);
		        BCL_vsub(vdy1, vyj, vy1);
		        BCL_vsub(vdy2, vyj, vy2);
		        BCL_vsub(vdy3, vyj, vy3);
	             
                        //d0 = 1.0 / (dx0 * dx0 + dy0 * dy0);
		        BCL_vmul(vd0, vdx0, vdx0);
		        BCL_vmul(vd1, vdx1, vdx1);
		        BCL_vmul(vd2, vdx2, vdx2);
		        BCL_vmul(vd3, vdx3, vdx3);
                        
                        BCL_vmac(vd0, vdy0, vdy0);
                        BCL_vmac(vd1, vdy1, vdy1);
                        BCL_vmac(vd2, vdy2, vdy2);
                        BCL_vmac(vd3, vdy3, vdy3);
                        
                        BCL_imaskz_vrcp(vd0, ik0); 
                        BCL_imaskz_vrcp(vd1, ik1); 
                        BCL_imaskz_vrcp(vd2, ik2); 
                        BCL_imaskz_vrcp(vd3, ik3); 
		     
                        //tfx0 += dx0 * d0;
                        BCL_vmac(vtfx0, vdx0, vd0);
                        BCL_vmac(vtfx1, vdx1, vd1);
                        BCL_vmac(vtfx2, vdx2, vd2);
                        BCL_vmac(vtfx3, vdx3, vd3);

                        //tfy0 += dy0 * d0;
                        BCL_vmac(vtfy0, vdy0, vd0);
                        BCL_vmac(vtfy1, vdy1, vd1);
                        BCL_vmac(vtfy2, vdy2, vd2);
                        BCL_vmac(vtfy3, vdy3, vd3);
                     }
                  }
/*
 *                Upper part  
 *
 */ 
                  chunksize = (M - i - UR*VLEN)/nthreads; 

                  for(j = (i+UR*VLEN)+id*chunksize; j < (i+UR*VLEN)+(id+1)*chunksize; j++)
                  {
                     register VTYPE vxj;
                     register VTYPE vyj;
		  
                     BCL_vbcast(vxj, blasX+j);
                     BCL_vbcast(vyj, blasY+j);
                     //dx0 = xj - x0;
		     BCL_vsub(vdx0, vxj, vx0);
		     BCL_vsub(vdx1, vxj, vx1);
		     BCL_vsub(vdx2, vxj, vx2);
		     BCL_vsub(vdx3, vxj, vx3);
	          
                     // dy0 = yj - y0;
		     BCL_vsub(vdy0, vyj, vy0);
		     BCL_vsub(vdy1, vyj, vy1);
		     BCL_vsub(vdy2, vyj, vy2);
		     BCL_vsub(vdy3, vyj, vy3);
	             
                     //d0 = 1.0 / (dx0 * dx0 + dy0 * dy0);
		     BCL_vmul(vd0, vdx0, vdx0);
		     BCL_vmul(vd1, vdx1, vdx1);
		     BCL_vmul(vd2, vdx2, vdx2);
		     BCL_vmul(vd3, vdx3, vdx3);
                     
                     BCL_vmac(vd0, vdy0, vdy0);
                     BCL_vmac(vd1, vdy1, vdy1);
                     BCL_vmac(vd2, vdy2, vdy2);
                     BCL_vmac(vd3, vdy3, vdy3);
                     
                     BCL_vrcp(vd0); 
                     BCL_vrcp(vd1); 
                     BCL_vrcp(vd2); 
                     BCL_vrcp(vd3); 
                  
                     //tfx0 += dx0 * d0;
                     BCL_vmac(vtfx0, vdx0, vd0);
                     BCL_vmac(vtfx1, vdx1, vd1);
                     BCL_vmac(vtfx2, vdx2, vd2);
                     BCL_vmac(vtfx3, vdx3, vd3);

                     //tfy0 += dy0 * d0;
                     BCL_vmac(vtfy0, vdy0, vd0);
                     BCL_vmac(vtfy1, vdy1, vd1);
                     BCL_vmac(vtfy2, vdy2, vd2);
                     BCL_vmac(vtfy3, vdy3, vd3);
                  }
/*
 *                cleanup 
 */
                  #pragma omp single nowait 
                  {
                     for (j=(i+UR*VLEN)+nthreads*chunksize; j < M; j++)
                     {
                        register VTYPE vxj; 
                        register VTYPE vyj;
		  
                        BCL_vbcast(vxj, blasX+j);
                        BCL_vbcast(vyj, blasY+j);
                        //dx0 = xj - x0;
		        BCL_vsub(vdx0, vxj, vx0);
		        BCL_vsub(vdx1, vxj, vx1);
		        BCL_vsub(vdx2, vxj, vx2);
		        BCL_vsub(vdx3, vxj, vx3);
	          
                        // dy0 = yj - y0;
		        BCL_vsub(vdy0, vyj, vy0);
		        BCL_vsub(vdy1, vyj, vy1);
		        BCL_vsub(vdy2, vyj, vy2);
		        BCL_vsub(vdy3, vyj, vy3);
	             
                        //d0 = 1.0 / (dx0 * dx0 + dy0 * dy0);
		        BCL_vmul(vd0, vdx0, vdx0);
		        BCL_vmul(vd1, vdx1, vdx1);
		        BCL_vmul(vd2, vdx2, vdx2);
		        BCL_vmul(vd3, vdx3, vdx3);
                        
                        BCL_vmac(vd0, vdy0, vdy0);
                        BCL_vmac(vd1, vdy1, vdy1);
                        BCL_vmac(vd2, vdy2, vdy2);
                        BCL_vmac(vd3, vdy3, vdy3);
                        
                        BCL_vrcp(vd0); 
                        BCL_vrcp(vd1); 
                        BCL_vrcp(vd2); 
                        BCL_vrcp(vd3); 
		     
                        //tfx0 += dx0 * d0;
                        BCL_vmac(vtfx0, vdx0, vd0);
                        BCL_vmac(vtfx1, vdx1, vd1);
                        BCL_vmac(vtfx2, vdx2, vd2);
                        BCL_vmac(vtfx3, vdx3, vd3);

                        //tfy0 += dy0 * d0;
                        BCL_vmac(vtfy0, vdy0, vd0);
                        BCL_vmac(vtfy1, vdy1, vd1);
                        BCL_vmac(vtfy2, vdy2, vd2);
                        BCL_vmac(vtfy3, vdy3, vd3);
                     }
                  }
/*
 *                Reduction without synchronization  
 */
                  {
                     register VTYPE tv; 
                     double *ptr = sumX + id*UR*VLEN;
                     
                     BCL_vldu(tv, ptr);
                     BCL_vadd(vtfx0, vtfx0, tv);
                     BCL_vstu(ptr, vtfx0);
                     
                     //fx1 += tfx1; 
                     BCL_vldu(tv, ptr+1*VLEN);
                     BCL_vadd(vtfx1, vtfx1, tv);
                     BCL_vstu(ptr+1*VLEN, vtfx1);
                     
                     BCL_vldu(tv, ptr+2*VLEN);
                     BCL_vadd(vtfx2, vtfx2, tv);
                     BCL_vstu(ptr+2*VLEN, vtfx2);

                     BCL_vldu(tv, ptr+3*VLEN);
                     BCL_vadd(vtfx3, vtfx3, tv);
                     BCL_vstu(ptr+3*VLEN, vtfx3);
                     
                     //fy1 += tfy1; 
                     ptr = (double*) sumY + id*UR*VLEN;
                     BCL_vldu(tv, ptr);
                     BCL_vadd(vtfy0, vtfy0, tv);
                     BCL_vstu(ptr, vtfy0);
                     
                     BCL_vldu(tv, ptr+1*VLEN);
                     BCL_vadd(vtfy1, vtfy1, tv);
                     BCL_vstu(ptr+1*VLEN, vtfy1);
                     
                     BCL_vldu(tv, ptr+2*VLEN);
                     BCL_vadd(vtfy2, vtfy2, tv);
                     BCL_vstu(ptr+2*VLEN, vtfy2);

                     BCL_vldu(tv, ptr+3*VLEN);
                     BCL_vadd(vtfy3, vtfy3, tv);
                     BCL_vstu(ptr+3*VLEN, vtfy3);
                  }
               }
/*
 *             Out of parallel region 
 */
/*
 *             sum the results up: 
 *             FIXME: use intrinsic to speed it up 
 */
               for(m=1; m < NUMOFTHREADS; m++)
               {
                  for (n=0; n < UR*VLEN; n++)
                  {
                     sumX[n] += sumX[m*UR*VLEN+n];
                     sumY[n] += sumY[m*UR*VLEN+n];
                  }
               }


               double *ptr = &sumX[0]; 
               VTYPE tvsum;
               
               
               BCL_vldu(tvsum, ptr);
               BCL_vstu(pb_X+ind, tvsum); 
               
               BCL_vldu(tvsum, ptr+1*VLEN);
               BCL_vstu(pb_X+ind+VLEN, tvsum); 
               
               BCL_vldu(tvsum, ptr+2*VLEN);
               BCL_vstu(pb_X+ind+2*VLEN, tvsum);

               BCL_vldu(tvsum, ptr+3*VLEN);
               BCL_vstu(pb_X+ind+3*VLEN, tvsum); 
               
               // Y  
               ptr = &sumY[0]; 
               BCL_vldu(tvsum, ptr);
               BCL_vstu(pb_Y+ind, tvsum); 
               
               BCL_vldu(tvsum, ptr+VLEN);
               BCL_vstu(pb_Y+ind+VLEN, tvsum); 
               
               BCL_vldu(tvsum, ptr+2*VLEN);
               BCL_vstu(pb_Y+ind+2*VLEN, tvsum);

               BCL_vldu(tvsum, ptr+3*VLEN);
               BCL_vstu(pb_Y+ind+3*VLEN, tvsum); 
            }

            // connected nodes 
	    for(INDEXTYPE i = b * BATCHSIZE; i < (b + 1) * BATCHSIZE; i += 1)
            {   
               int ind = i-b*BATCHSIZE;
               VALUETYPE pbX=0.0, pbY=0.0;
               VALUETYPE dist, distX, distY;
               for(INDEXTYPE j = graph.rowptr[i]; j < graph.rowptr[i+1]; j += 1)
               {
                  int v = graph.colids[j];
                  distX = blasX[v] - blasX[i];
                  distY = blasY[v] - blasY[i];
                  dist = (distX * distX + distY * distY);
                  dist = sqrt(dist) + 1.0 / dist;
                  pbX += distX * dist;
                  pbY += distY * dist;
               }
               pb_X[ind] = pbX - pb_X[ind];
               pb_Y[ind] = pbY - pb_Y[ind];
            }
	    /*printf("After:\n");
            for(INDEXTYPE i = b * BATCHSIZE; i < (b + 1) * BATCHSIZE; i++)
            {
               printf("%d = %lf,", i, pb_X[i-b*BATCHSIZE]);
            }
	    */
	    for(INDEXTYPE i = b * BATCHSIZE; i < (b + 1) * BATCHSIZE; i += 1)
            {
               int ind = i-b*BATCHSIZE;
               VALUETYPE d;
	       d = (pb_X[ind] * pb_X[ind] + pb_Y[ind] * pb_Y[ind]);
	       ENERGY += d ;
	       
               d = STEP / sqrt(d);
	       blasX[i] += pb_X[ind] * d;
               blasY[i] += pb_Y[ind] * d;
	       
               pb_X[ind] = 0;
	       pb_Y[ind] = 0;    
            }
         }
/*
 *    cleanup 
 */
			INDEXTYPE cleanup = (graph.rows/BATCHSIZE) * BATCHSIZE;
			#pragma omp parallel for schedule(dynamic) 
			for(INDEXTYPE i = cleanup; i < graph.rows; i += 1){
                                INDEXTYPE ind = i- cleanup;
                                VALUETYPE fx = 0, fy = 0, distX, distY, dist, dist2;
                                for(INDEXTYPE j = graph.rowptr[i]; j < graph.rowptr[i+1]; j += 1){
                                        int v = graph.colids[j];
                                        distX = blasX[v] - blasX[i];
                                        distY = blasY[v] - blasY[i];
                                        dist = (distX * distX + distY * distY);
                                        dist = sqrt(dist) + 1.0 / dist;
                                        pb_X[ind] += distX * dist;
                                        pb_Y[ind] += distY * dist;
                                }
                                for(INDEXTYPE j = 0; j < i; j += 1){
                                        distX = blasX[j] - blasX[i];
                                        distY = blasY[j] - blasY[i];
                                        dist2 = 1.0 / (distX * distX + distY * distY);
                                        fx += distX * dist2;
                                        fy += distY * dist2;
                                }
                                for(INDEXTYPE j = i+1; j < graph.rows; j += 1){
                                        distX = blasX[j] - blasX[i];
                                        distY = blasY[j] - blasY[i];
                                        dist2 = 1.0 / (distX * distX + distY * distY);
                                        fx += distX * dist2;
                                        fy += distY * dist2;
                                }
                                pb_X[ind] = pb_X[ind] - fx;
                                pb_Y[ind] = pb_Y[ind] - fy;
                        }	
			for(INDEXTYPE i = cleanup; i < graph.rows; i += 1){
                                int ind = i-cleanup;
                                double dist = (pb_X[ind]*pb_X[ind] + pb_Y[ind]*pb_Y[ind]);
                                ENERGY += dist;
				dist = STEP / sqrt(dist);
				blasX[i] += pb_X[ind] * dist;
                                blasY[i] += pb_Y[ind] * dist;
				pb_X[ind] = pb_Y[ind] = 0;
                        }
                        STEP = STEP * 0.999;
                        LOOP++;
                }
                end = omp_get_wtime();
         #if NO_FWRITE == 0
                cout << "Efficientunroll Minibatch Size:" << BATCHSIZE  << endl;
                cout << "Efficientunroll Minbatch Energy:" << ENERGY << endl;
                cout << "Efficientunroll Minibatch Parallel Wall time required:" << end - start << endl;
                writeToFileEFF("EFFUR4-VEC-NOSYNC"+ to_string(BATCHSIZE)+"PARAOUT" + to_string(LOOP));
         #endif
                result.push_back(ENERGY);
                result.push_back(end - start);
/*
 *              NOTE: free allocated memory... Why not done for other mem???
 */
                free(sumX);
                free(sumY);
                return result;
        }
#elif defined (MDIM_VEC_UR4_MASK_NOSYNC)
/*
 * using simd.h ... to generalize  
 */
   vector<VALUETYPE> newalgo::EfficientVersionMdim(INDEXTYPE ITERATIONS, 
         INDEXTYPE NUMOFTHREADS, INDEXTYPE BATCHSIZE)
   {
      INDEXTYPE LOOP = 0;
      VALUETYPE start, end, ENERGY, ENERGY0, *pb_X, *pb_Y;
      VALUETYPE STEP = 1.0;
      VALUETYPE *sumX, *sumY;
       
      vector<VALUETYPE> result;
      pb_X = static_cast<VALUETYPE *> (::operator new (sizeof(VALUETYPE[BATCHSIZE])));
      pb_Y = static_cast<VALUETYPE *> (::operator new (sizeof(VALUETYPE[BATCHSIZE])));
      ENERGY0 = ENERGY = numeric_limits<VALUETYPE>::max();

/*
 *    NOTE: to avoid overflow the stack, we allocate memory in heap to 
 *    manage reduction .... 
 *    FIXME: possible to combine it with pb_X and pb_Y???? 
 *           size = NUMOFTHREADS * UNROLL * VLEN
 *    solved the stack overflow by this way 
 *    a test case 
 */
      #define UR 4  /* unroll factor = 4 */
      sumX = static_cast<VALUETYPE *> 
               (::operator new (sizeof(VALUETYPE[NUMOFTHREADS*UR*VLEN])));
      sumY = static_cast<VALUETYPE *> 
               (::operator new (sizeof(VALUETYPE[NUMOFTHREADS*UR*VLEN])));

      omp_set_num_threads(NUMOFTHREADS);
/*
 *    time included initDFS 
 */
      start = omp_get_wtime();

      initDFS();
      
      while(LOOP < ITERATIONS)
      {
         ENERGY0 = ENERGY;
         ENERGY = 0;
         for(int i = 0; i < BATCHSIZE; i++)
         {
            pb_X[i] = pb_Y[i] = 0;
         }
         
	 for(INDEXTYPE b = 0; b < (graph.rows / BATCHSIZE); b += 1)
         {
	    for(INDEXTYPE i = b * BATCHSIZE; i < (b + 1) * BATCHSIZE; i += UR*VLEN)
            {
	       int ind = i-b*BATCHSIZE;
               int m, n, k;
               register VTYPE vx0, vx1, vx2, vx3;
               register VTYPE vy0, vy1, vy2, vy3;
/*
 *             to minimize critical section, we are using extra memory here
 *             FIXME: use intrinsic to optimize it, like: XoR  
 */
               for (m=0; m < NUMOFTHREADS*UR*VLEN; m++)
                  sumX[m] = sumY[m] = 0.0;

	       BCL_vldu(vx0, blasX + i);
	       BCL_vldu(vx1, blasX + i+VLEN);
	       BCL_vldu(vx2, blasX + i+2*VLEN);
	       BCL_vldu(vx3, blasX + i+3*VLEN);

	       BCL_vldu(vy0, blasY + i);
	       BCL_vldu(vy1, blasY + i+1*VLEN);
	       BCL_vldu(vy2, blasY + i+2*VLEN);
	       BCL_vldu(vy3, blasY + i+3*VLEN);
/*
 *             starting of parallel region 
 */
               #pragma omp parallel shared(sumX, sumY)
               {
                  int id, nthreads; 
                  int chunksize;
                  int j; 
                  VTYPE vdx0, vdx1, vdx2, vdx3;
                  VTYPE vdy0, vdy1, vdy2, vdy3;
                  VTYPE vd0, vd1, vd2, vd3;
                  
                  VTYPE vtfx0, vtfx1, vtfx2, vtfx3;
                  VTYPE vtfy0, vtfy1, vtfy2, vtfy3;

                  const int M = graph.rows; 
                  id = omp_get_thread_num();
                  nthreads = omp_get_num_threads();
                  
                  // init temp sum 
		  BCL_vzero(vtfx0); BCL_vzero(vtfy0);
		  BCL_vzero(vtfx1); BCL_vzero(vtfy1);
		  BCL_vzero(vtfx2); BCL_vzero(vtfy2);
		  BCL_vzero(vtfx3); BCL_vzero(vtfy3);
/*
 *                j is up to i... lower up case  
 */
                  chunksize = M/nthreads; 
                  
                  for(j = id*chunksize; j < (id+1)*chunksize; j++)
                  {
                     const int n = j-i;
                     register VTYPE vxj; 
                     register VTYPE vyj;
                     unsigned int ik0, ik1, ik2, ik3; 
                     const unsigned int AONE = (1<<VLEN)-1; // 255 
		  
                     BCL_vbcast(vxj, blasX+j);
                     BCL_vbcast(vyj, blasY+j);
                     
                     ik0 = ( n >= 0 && n < VLEN ) ? 
                              (AONE & (~(1 << (n%VLEN)))) : AONE; 
                     ik1 = ( n >= VLEN && n < 2*VLEN )? 
                              (AONE & ~(1 << n%VLEN))     : AONE; 
                     ik2 = ( n >=2*VLEN && n < 3*VLEN )? 
                              (AONE & ~(1 << n%VLEN))     : AONE; 
                     ik3 = ( n >=3*VLEN && n < 4*VLEN )? 
                              (AONE & ~(1 << n%VLEN))     : AONE; 
                     
                     //dx0 = xj - x0;
		     BCL_vsub(vdx0, vxj, vx0);
		     BCL_vsub(vdx1, vxj, vx1);
		     BCL_vsub(vdx2, vxj, vx2);
		     BCL_vsub(vdx3, vxj, vx3);
	          
                     // dy0 = yj - y0;
		     BCL_vsub(vdy0, vyj, vy0);
		     BCL_vsub(vdy1, vyj, vy1);
		     BCL_vsub(vdy2, vyj, vy2);
		     BCL_vsub(vdy3, vyj, vy3);
	             
                     //d0 = 1.0 / (dx0 * dx0 + dy0 * dy0);
		     BCL_vmul(vd0, vdx0, vdx0);
		     BCL_vmul(vd1, vdx1, vdx1);
		     BCL_vmul(vd2, vdx2, vdx2);
		     BCL_vmul(vd3, vdx3, vdx3);
                     
                     BCL_vmac(vd0, vdy0, vdy0);
                     BCL_vmac(vd1, vdy1, vdy1);
                     BCL_vmac(vd2, vdy2, vdy2);
                     BCL_vmac(vd3, vdy3, vdy3);
                     
                     BCL_imaskz_vrcp(vd0, ik0);
                     BCL_imaskz_vrcp(vd1, ik1);
                     BCL_imaskz_vrcp(vd2, ik2);
                     BCL_imaskz_vrcp(vd3, ik3);
                     //tfx0 += dx0 * d0;
                     BCL_vmac(vtfx0, vdx0, vd0);
                     BCL_vmac(vtfx1, vdx1, vd1);
                     BCL_vmac(vtfx2, vdx2, vd2);
                     BCL_vmac(vtfx3, vdx3, vd3);

                     //tfy0 += dy0 * d0;
                     BCL_vmac(vtfy0, vdy0, vd0);
                     BCL_vmac(vtfy1, vdy1, vd1);
                     BCL_vmac(vtfy2, vdy2, vd2);
                     BCL_vmac(vtfy3, vdy3, vd3);
                  }
/*
 *                cleanup 
 */
                  #pragma omp single nowait 
                  {
                     for (j=nthreads*chunksize; j < M; j++)
                     {
                        const int n = j-i;
                        register VTYPE vxj;
                        register VTYPE vyj;
                        unsigned int ik0, ik1, ik2, ik3; 
                        const unsigned int AONE = (1<<VLEN)-1; // 255 
		 
                        BCL_vbcast(vxj, blasX+j);
                        BCL_vbcast(vyj, blasY+j);

                        ik0 = ( n >= 0 && n < VLEN ) ? 
                              (AONE & (~(1 << (n%VLEN)))) : AONE; 
                        ik1 = ( n >= VLEN && n < 2*VLEN )? 
                              (AONE & ~(1 << n%VLEN))     : AONE; 
                        ik2 = ( n >=2*VLEN && n < 3*VLEN )? 
                              (AONE & ~(1 << n%VLEN))     : AONE; 
                        ik3 = ( n >=3*VLEN && n < 4*VLEN )? 
                              (AONE & ~(1 << n%VLEN))     : AONE; 
		  
                        //dx0 = xj - x0;
		        BCL_vsub(vdx0, vxj, vx0);
		        BCL_vsub(vdx1, vxj, vx1);
		        BCL_vsub(vdx2, vxj, vx2);
		        BCL_vsub(vdx3, vxj, vx3);
	          
                        // dy0 = yj - y0;
		        BCL_vsub(vdy0, vyj, vy0);
		        BCL_vsub(vdy1, vyj, vy1);
		        BCL_vsub(vdy2, vyj, vy2);
		        BCL_vsub(vdy3, vyj, vy3);
	             
                        //d0 = 1.0 / (dx0 * dx0 + dy0 * dy0);
		        BCL_vmul(vd0, vdx0, vdx0);
		        BCL_vmul(vd1, vdx1, vdx1);
		        BCL_vmul(vd2, vdx2, vdx2);
		        BCL_vmul(vd3, vdx3, vdx3);
                        
                        BCL_vmac(vd0, vdy0, vdy0);
                        BCL_vmac(vd1, vdy1, vdy1);
                        BCL_vmac(vd2, vdy2, vdy2);
                        BCL_vmac(vd3, vdy3, vdy3);
                        
                        BCL_imaskz_vrcp(vd0, ik0); 
                        BCL_imaskz_vrcp(vd1, ik1); 
                        BCL_imaskz_vrcp(vd2, ik2); 
                        BCL_imaskz_vrcp(vd3, ik3); 
		     
                        //tfx0 += dx0 * d0;
                        BCL_vmac(vtfx0, vdx0, vd0);
                        BCL_vmac(vtfx1, vdx1, vd1);
                        BCL_vmac(vtfx2, vdx2, vd2);
                        BCL_vmac(vtfx3, vdx3, vd3);

                        //tfy0 += dy0 * d0;
                        BCL_vmac(vtfy0, vdy0, vd0);
                        BCL_vmac(vtfy1, vdy1, vd1);
                        BCL_vmac(vtfy2, vdy2, vd2);
                        BCL_vmac(vtfy3, vdy3, vd3);
                     }
                  }
/*
 *                Reduction without synchronization  
 */
                  {
                     register VTYPE tv; 
                     double *ptr = sumX + id*UR*VLEN;
                     
                     BCL_vldu(tv, ptr);
                     BCL_vadd(vtfx0, vtfx0, tv);
                     BCL_vstu(ptr, vtfx0);
                     
                     BCL_vldu(tv, ptr+1*VLEN);
                     BCL_vadd(vtfx1, vtfx1, tv);
                     BCL_vstu(ptr+1*VLEN, vtfx1);
                     
                     BCL_vldu(tv, ptr+2*VLEN);
                     BCL_vadd(vtfx2, vtfx2, tv);
                     BCL_vstu(ptr+2*VLEN, vtfx2);

                     BCL_vldu(tv, ptr+3*VLEN);
                     BCL_vadd(vtfx3, vtfx3, tv);
                     BCL_vstu(ptr+3*VLEN, vtfx3);
                     
                     //fy1 += tfy1; 
                     ptr = (double*) sumY + id*UR*VLEN;
                     BCL_vldu(tv, ptr);
                     BCL_vadd(vtfy0, vtfy0, tv);
                     BCL_vstu(ptr, vtfy0);
                     
                     BCL_vldu(tv, ptr+1*VLEN);
                     BCL_vadd(vtfy1, vtfy1, tv);
                     BCL_vstu(ptr+1*VLEN, vtfy1);
                     
                     BCL_vldu(tv, ptr+2*VLEN);
                     BCL_vadd(vtfy2, vtfy2, tv);
                     BCL_vstu(ptr+2*VLEN, vtfy2);

                     BCL_vldu(tv, ptr+3*VLEN);
                     BCL_vadd(vtfy3, vtfy3, tv);
                     BCL_vstu(ptr+3*VLEN, vtfy3);
                  }
               }
/*
 *             Out of parallel region 
 */
/*
 *             sum the results up: 
 *             FIXME: use intrinsic to speed it up 
 */
               for(m=1; m < NUMOFTHREADS; m++)
               {
                  for (n=0; n < UR*VLEN; n++)
                  {
                     sumX[n] += sumX[m*UR*VLEN+n];
                     sumY[n] += sumY[m*UR*VLEN+n];
                  }
               }


               double *ptr = &sumX[0]; 
               VTYPE tvsum;
               
               
               BCL_vldu(tvsum, ptr);
               BCL_vstu(pb_X+ind, tvsum); 
               
               BCL_vldu(tvsum, ptr+1*VLEN);
               BCL_vstu(pb_X+ind+VLEN, tvsum); 
               
               BCL_vldu(tvsum, ptr+2*VLEN);
               BCL_vstu(pb_X+ind+2*VLEN, tvsum);

               BCL_vldu(tvsum, ptr+3*VLEN);
               BCL_vstu(pb_X+ind+3*VLEN, tvsum); 
               
               // Y  
               ptr = &sumY[0]; 
               BCL_vldu(tvsum, ptr);
               BCL_vstu(pb_Y+ind, tvsum); 
               
               BCL_vldu(tvsum, ptr+VLEN);
               BCL_vstu(pb_Y+ind+VLEN, tvsum); 
               
               BCL_vldu(tvsum, ptr+2*VLEN);
               BCL_vstu(pb_Y+ind+2*VLEN, tvsum);

               BCL_vldu(tvsum, ptr+3*VLEN);
               BCL_vstu(pb_Y+ind+3*VLEN, tvsum); 
            }

            // connected nodes 
	    for(INDEXTYPE i = b * BATCHSIZE; i < (b + 1) * BATCHSIZE; i += 1)
            {   
               int ind = i-b*BATCHSIZE;
               VALUETYPE pbX=0.0, pbY=0.0;
               VALUETYPE dist, distX, distY;
               for(INDEXTYPE j = graph.rowptr[i]; j < graph.rowptr[i+1]; j += 1)
               {
                  int v = graph.colids[j];
                  distX = blasX[v] - blasX[i];
                  distY = blasY[v] - blasY[i];
                  dist = (distX * distX + distY * distY);
                  dist = sqrt(dist) + 1.0 / dist;
                  pbX += distX * dist;
                  pbY += distY * dist;
               }
               pb_X[ind] = pbX - pb_X[ind];
               pb_Y[ind] = pbY - pb_Y[ind];
            }
	    /*printf("After:\n");
            for(INDEXTYPE i = b * BATCHSIZE; i < (b + 1) * BATCHSIZE; i++)
            {
               printf("%d = %lf,", i, pb_X[i-b*BATCHSIZE]);
            }
	    */
	    for(INDEXTYPE i = b * BATCHSIZE; i < (b + 1) * BATCHSIZE; i += 1)
            {
               int ind = i-b*BATCHSIZE;
               VALUETYPE d;
	       d = (pb_X[ind] * pb_X[ind] + pb_Y[ind] * pb_Y[ind]);
	       ENERGY += d ;
	       
               d = STEP / sqrt(d);
	       blasX[i] += pb_X[ind] * d;
               blasY[i] += pb_Y[ind] * d;
	       
               pb_X[ind] = 0;
	       pb_Y[ind] = 0;    
            }
         }
/*
 *    cleanup 
 */
			INDEXTYPE cleanup = (graph.rows/BATCHSIZE) * BATCHSIZE;
			#pragma omp parallel for schedule(dynamic) 
			for(INDEXTYPE i = cleanup; i < graph.rows; i += 1){
                                INDEXTYPE ind = i- cleanup;
                                VALUETYPE fx = 0, fy = 0, distX, distY, dist, dist2;
                                for(INDEXTYPE j = graph.rowptr[i]; j < graph.rowptr[i+1]; j += 1){
                                        int v = graph.colids[j];
                                        distX = blasX[v] - blasX[i];
                                        distY = blasY[v] - blasY[i];
                                        dist = (distX * distX + distY * distY);
                                        dist = sqrt(dist) + 1.0 / dist;
                                        pb_X[ind] += distX * dist;
                                        pb_Y[ind] += distY * dist;
                                }
                                for(INDEXTYPE j = 0; j < i; j += 1){
                                        distX = blasX[j] - blasX[i];
                                        distY = blasY[j] - blasY[i];
                                        dist2 = 1.0 / (distX * distX + distY * distY);
                                        fx += distX * dist2;
                                        fy += distY * dist2;
                                }
                                for(INDEXTYPE j = i+1; j < graph.rows; j += 1){
                                        distX = blasX[j] - blasX[i];
                                        distY = blasY[j] - blasY[i];
                                        dist2 = 1.0 / (distX * distX + distY * distY);
                                        fx += distX * dist2;
                                        fy += distY * dist2;
                                }
                                pb_X[ind] = pb_X[ind] - fx;
                                pb_Y[ind] = pb_Y[ind] - fy;
                        }	
			for(INDEXTYPE i = cleanup; i < graph.rows; i += 1){
                                int ind = i-cleanup;
                                double dist = (pb_X[ind]*pb_X[ind] + pb_Y[ind]*pb_Y[ind]);
                                ENERGY += dist;
				dist = STEP / sqrt(dist);
				blasX[i] += pb_X[ind] * dist;
                                blasY[i] += pb_Y[ind] * dist;
				pb_X[ind] = pb_Y[ind] = 0;
                        }
                        STEP = STEP * 0.999;
                        LOOP++;
                }
                end = omp_get_wtime();
         #if NO_FWRITE == 0
                cout << "Efficientunroll Minibatch Size:" << BATCHSIZE  << endl;
                cout << "Efficientunroll Minbatch Energy:" << ENERGY << endl;
                cout << "Efficientunroll Minibatch Parallel Wall time required:" << end - start << endl;
                writeToFileEFF("EFFUR4-VEC-MASK-NOSYNC"+ to_string(BATCHSIZE)+"PARAOUT" + to_string(LOOP));
         #endif
                result.push_back(ENERGY);
                result.push_back(end - start);
/*
 *              NOTE: free allocated memory... Why not done for other mem???
 */
                free(sumX);
                free(sumY);
                return result;
        }

#endif




