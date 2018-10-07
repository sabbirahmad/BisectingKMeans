import java.awt.Dimension;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.FileReader;
import java.io.IOException;
import java.io.ObjectOutputStream;
import java.io.OutputStream;
import java.io.PrintWriter;
import java.lang.reflect.Array;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.PriorityQueue;
import java.util.Random;
import java.util.Scanner;
import java.util.concurrent.CopyOnWriteArrayList;

public class BisectingKMeans {
	private final int NIteration=10;
	private double B=30;
	private String fileName;
	private int dimension=2;
	private int dataCount;
	private int classCount;
	private int K=12;
	private double SSError=0;
	
	private Random random;
	
	//CopyOnWriteArrayList<Cluster> clusters;
	CopyOnWriteArrayList<ArrayList<Double>> data;
	CopyOnWriteArrayList<Integer> classList;
	
	ArrayList<ArrayList<Double>> initialCentroidForKmeans;
	ArrayList<ArrayList<Double>> initialCentroidRandom;
	
	PriorityQueue<Cluster> PQ=new PriorityQueue<Cluster>(20, new Comparator<Cluster>(){
		public int compare(Cluster c1, Cluster c2){
			if((c2.getSE()-c1.getSE())>0)
				return 1;
			if((c2.getSE()-c1.getSE())<0)
				return -1;
			else
				return 0;
		}
	});
	
	public BisectingKMeans(String fileName) throws IOException {
		this.fileName=fileName;
		
		random=new Random();
		//clusters=new CopyOnWriteArrayList<Cluster>();
		data=new CopyOnWriteArrayList<ArrayList<Double>>();
		classList=new CopyOnWriteArrayList<Integer>();
		initialCentroidForKmeans=new ArrayList<ArrayList<Double>>();
		initialCentroidRandom=new ArrayList<ArrayList<Double>>();
		
		readData(fileName);
		
		BiKMeans(data);
		
		System.out.println("End");
	}
	
	private void readData(String fname) throws FileNotFoundException{
		int i;
        Scanner in = new Scanner(new FileReader(new File(fileName)));
        
        //dimension=in.nextInt();
        //classCount=in.nextInt();
        //dataCount=in.nextInt();
        //int j=0;
        
        
        while(in.hasNext()){
        	ArrayList<Double> temp=new ArrayList<Double>();
        	//System.out.println("Line: "+j);
             for(i=0;i<dimension;i++){
            	 //System.out.println("Line elem: "+i);
            	 temp.add(in.nextDouble());
             }
             //classList.add(in.nextInt());  //if no class is given, comment this out
             data.add(temp);
             //j++;
        }
        in.close();
        //System.out.println("input taken");
	}
	
	private void BiKMeans(CopyOnWriteArrayList<ArrayList<Double>> data) throws FileNotFoundException, IOException{
		ArrayList<Double> initialCen=calculateCentroid(data);
		Cluster initialCluster=new Cluster(data,initialCen);
		double initialSE=calculateSE(initialCluster);
		initialCluster.setSE(initialSE);
		//double initialSSE=initialCluster.getSE();
		
		SSError=initialCluster.getSE();
		
		System.out.println("Initial centroid: "+initialCluster.getCentroid());
		System.out.println("Initial SE: "+initialCluster.getSE());
		
		//twoMeans(initialCluster,initialSSE);
		PQ.add(initialCluster);
		
		for(int i=1;i<K;i++){
			Cluster cls=PQ.poll();
			twoMeans(cls);
			//System.out.println("SSE-"+i+": "+SSError);
		}
		
		PrintWriter out=new PrintWriter("cen.txt");
		
		int i=0;
		for (Cluster cl : PQ) {
			System.out.println("Cluster: "+i);
			System.out.println("\tC: "+cl.getCentroid());
			//System.out.println("\tSE: "+cl.getSE());
			i++;
			initialCentroidForKmeans.add(cl.getCentroid());//set initial centroids for kmean
			ArrayList<Double> cent=cl.getCentroid();
			for(int j=0;j<dimension;j++){
				out.write(cent.get(j).toString()+" ");
			}
			out.write("\n");
		}
		out.close();
		System.out.println("Bisection SSE: "+SSError);
		
		/*//randomly initialize centroids
		for (i=0;i<K;i++){ //for all clusters randomly generate controids
			ArrayList<Double> cen=new ArrayList<Double>();
			for(int j=0;j<dimension;j++){
				cen.add(B*random.nextDouble());
				//cen2.add(B*random.nextDouble());
				//System.out.print(cen.get(j)+" ");
			}
			//System.out.println("");
			initialCentroidRandom.add(cen);
		}*/
		
		
		//run kmeans
		kMeans(data, initialCentroidForKmeans, K);
		
		//run kmeans with random
		//kMeans(data, initialCentroidRandom, K);
	}
	
	private void twoMeans(Cluster cluster){
		Cluster cluster1=null;
		Cluster cluster2=null;
		
		double bClusterSE=cluster.getSE();
		double updatedSSE=SSError-bClusterSE;
		double prevSSE=updatedSSE;
		
		int itr;
		for(itr=0;itr<NIteration;itr++){
			
			ArrayList<Double> cen1=new ArrayList<Double>();
			ArrayList<Double> cen2=new ArrayList<Double>();
			int i,j;
			//initialization of 2 centroids of bisection
			for(i=0;i<dimension;i++){
				cen1.add(B*random.nextDouble());
				//cen2.add(B*random.nextDouble());
			}
			ArrayList<Double> clCen=cluster.getCentroid();
			for(i=0;i<dimension;i++){
				cen2.add(2*clCen.get(i)-cen1.get(i));
			}
			
			
			CopyOnWriteArrayList<ArrayList<Double>> tempData=new CopyOnWriteArrayList<ArrayList<Double>>(cluster.getData());
			
			CopyOnWriteArrayList<ArrayList<Double>> c1=new CopyOnWriteArrayList<ArrayList<Double>>();
			CopyOnWriteArrayList<ArrayList<Double>> c2=new CopyOnWriteArrayList<ArrayList<Double>>();
			
			double dis1,dis2;
			int size=tempData.size();
			while(true){
				//System.out.println("while");
				//keep previous centroids
				ArrayList<Double> prevCen1=new ArrayList<Double>(cen1);
				ArrayList<Double> prevCen2=new ArrayList<Double>(cen2);
				//clear 2 clusters to start again
				c1.clear();
				c2.clear();
				
				//set points in cluster
				for(i=0;i<size;i++){
					dis1=dis2=0;
					ArrayList<Double> temp=tempData.get(i);
					for(j=0;j<dimension;j++){
						dis1+=(temp.get(j)-cen1.get(j))*(temp.get(j)-cen1.get(j));
						dis2+=(temp.get(j)-cen2.get(j))*(temp.get(j)-cen2.get(j));
					}
					if(dis2>dis1){
						c1.add(temp);
					}
					else{
						c2.add(temp);
					}
				}
				
				//update centroid
				int size1=c1.size();
				int size2=c2.size();
				ArrayList<Double> newCen=new ArrayList<Double>();
				//initial newcentroid all 0
				for(i=0;i<dimension;i++){
					newCen.add(0.0);
				}
				//update centroid for c1
				for(i=0;i<size1;i++){
					ArrayList<Double> temp=c1.get(i);
					for(j=0;j<dimension;j++){
						newCen.set(j, (newCen.get(j)+temp.get(j)));
					}
				}
				for(i=0;i<dimension;i++){
					cen1.set(i, newCen.get(i)/size1);
				}
				
				//update centroid for c2
				//initial newcentroid all 0
				newCen.clear();
				for(i=0;i<dimension;i++){
					newCen.add(0.0);
				}
				
				for(i=0;i<size2;i++){
					ArrayList<Double> temp=c2.get(i);
					for(j=0;j<dimension;j++){
						newCen.set(j, (newCen.get(j)+temp.get(j)));
					}
				}
				for(i=0;i<dimension;i++){
					cen2.set(i, newCen.get(i)/size2);
				}
				
				//check breaking condition
				dis1=dis2=0;
				for(i=0;i<dimension;i++){
					dis1+=(prevCen1.get(i)-cen1.get(i))*(prevCen1.get(i)-cen1.get(i));
					dis2+=(prevCen2.get(i)-cen2.get(i))*(prevCen2.get(i)-cen2.get(i));
				}
				dis1=Math.sqrt(dis1);
				dis2=Math.sqrt(dis2);
				if(dis1<.001 && dis2<.001){//change of centroid is very minimal
					break;
				}
			}
			
			Cluster cl1=new Cluster(c1,cen1);
			Cluster cl2=new Cluster(c2,cen2);
			
			double se1=calculateSE(cl1);
			double se2=calculateSE(cl2);
			cl1.setSE(se1);
			cl2.setSE(se2);
			
			
			if(itr==0){
				cluster1= new Cluster(cl1);
				cluster2= new Cluster(cl2);
				prevSSE=updatedSSE+se1+se2;
			}
			
			else if((updatedSSE+se1+se2)<prevSSE){
				cluster1= new Cluster(cl1);
				cluster2= new Cluster(cl2);
				prevSSE=updatedSSE+se1+se2;
			}
			
			
			
		}
		//clusters.add(cluster1);
		//clusters.add(cluster2);
		PQ.add(cluster1);
		PQ.add(cluster2);
		SSError=prevSSE;
		
	}
	
	private void kMeans(CopyOnWriteArrayList<ArrayList<Double>> data, ArrayList<ArrayList<Double>> initialCentroid ,int k) throws FileNotFoundException{
		int i,j;
		ArrayList<ArrayList<Double>> prevCen=new ArrayList<ArrayList<Double>>(initialCentroid);
		ArrayList<Cluster> clusters=new ArrayList<Cluster>();
		for(i=0;i<k;i++){
			clusters.add(new Cluster());
			clusters.get(i).setCentroid(initialCentroid.get(i));
		}
		int size=data.size();
		double minDis=99999999.0;
		int minCluster=0;
		double dis;
		double sse=0;
		//int loop=0;
		while(true){
			//add data to clusters
			for(i=0;i<size;i++){
				minDis=99999999.0;
				minCluster=0;
				ArrayList<Double> temp=data.get(i);
				for(int c=0;c<k;c++){ //for each cluster
					dis=0;
					ArrayList<Double> cen=clusters.get(c).getCentroid();//get centroid of the cluster
					for(j=0;j<dimension;j++){
						dis+=((temp.get(j)-cen.get(j))*(temp.get(j)-cen.get(j)));
					}
					if(dis<minDis){ //check if less distance is found or not
						minDis=dis;
						minCluster=c;
					}
				}
				clusters.get(minCluster).getData().add(temp);
			}
			//update centroids
			for(i=0;i<k;i++){
				clusters.get(i).updateCentroid(dimension);
				double se=calculateSE(clusters.get(i));
				clusters.get(i).setSE(se);
			}
			
			//calculate SSE
			sse=0;
			for(i=0;i<k;i++){
				sse+=clusters.get(i).getSE();
			}
			//check centroid change
			double cenDis=0;
			for(i=0;i<k;i++){
				cenDis=0;
				for(j=0;j<dimension;j++){
					cenDis+=((clusters.get(i).getCentroid().get(j)-prevCen.get(i).get(j))*(clusters.get(i).getCentroid().get(j)-prevCen.get(i).get(j)));
				}
				cenDis=Math.sqrt(cenDis);
				if(cenDis>0.0001){//if distance is greater than this then other calculations are not necessary
					break;
				}
			}
			
			
			if(i==k){ //means all centers are unchanged
				break;
			}
			//update previous centroids
			prevCen.clear();
			for(i=0;i<k;i++){
				prevCen.add(clusters.get(i).getCentroid());
			}
			
			//System.out.println("loop: "+loop);
			//loop++;
		}
		
		PrintWriter out=new PrintWriter("kmeanCen.txt");
		System.out.println("K-Means result:");
		i=0;
		for (Cluster cl : clusters) {
			System.out.println("Cluster: "+i);
			System.out.println("\tC: "+cl.getCentroid());
			System.out.println("\tPoints: "+cl.getData().size());
			//System.out.println("\tSE: "+cl.getSE());
			i++;
			ArrayList<Double> cent=cl.getCentroid();
			for(j=0;j<dimension;j++){
				out.write(cent.get(j).toString()+" ");
			}
			out.write("\n");
		}
		out.close();
		
		System.out.println("K-means SSE: "+sse );
		
	}
	
	private double calculateSE(Cluster cluster){
		ArrayList<Double> cen=cluster.getCentroid();
		CopyOnWriteArrayList<ArrayList<Double>> tempData=cluster.getData();
		int size=tempData.size();
		int i,j;
		double error=0;
		for(i=0;i<size;i++){
			ArrayList<Double> temp=tempData.get(i);
			for(j=0;j<dimension;j++){
				error+=(temp.get(j)-cen.get(j))*(temp.get(j)-cen.get(j));
			}
		}
		
		return error;
	}
	
	private ArrayList<Double> calculateCentroid(CopyOnWriteArrayList<ArrayList<Double>> tData){
		ArrayList<Double> cen=new ArrayList<Double>();
		int i,j;
		for(i=0;i<dimension;i++){
			cen.add(0.0);
		}
		
		int size=tData.size();
		for(i=0;i<size;i++){
			ArrayList<Double> temp=tData.get(i);
			for(j=0;j<dimension;j++){
				cen.set(j, (cen.get(j)+temp.get(j)));
			}
		}
		
		for(j=0;j<dimension;j++){
			cen.set(j, (cen.get(j)/size));
		}
		
		return cen;
	}
	

	public static void main(String[] args) {
		// TODO Auto-generated method stub
		String fileName="//Users//ahmadsabbir//Documents//workspace//BisectingKMeans//resource//bisecting.txt";
		try {
			new BisectingKMeans(fileName);
		} catch (FileNotFoundException e) {
			System.out.println("File not found!!");
		} catch (IOException e) {
			System.out.println("Could not write file!!");
		}
	}

}

class Cluster{
	private double se;
	private ArrayList<Double> centroid;
	private CopyOnWriteArrayList<ArrayList<Double>> data;
	
	public Cluster(){
		se=0;
		centroid=new ArrayList<Double>();
		data=new CopyOnWriteArrayList<ArrayList<Double>>();
	}
	public Cluster(CopyOnWriteArrayList<ArrayList<Double>> pData, ArrayList<Double> cen){
		centroid=new ArrayList<Double>(cen);
		data=new CopyOnWriteArrayList<ArrayList<Double>>(pData);
	}
	public Cluster(Cluster cl){
		centroid=new ArrayList<Double>(cl.getCentroid());
		data=new CopyOnWriteArrayList<ArrayList<Double>>(cl.getData());
		se=cl.getSE();
	}
	public CopyOnWriteArrayList<ArrayList<Double>> getData(){
		return data;
	}
	public ArrayList<Double> getCentroid(){
		return centroid;
	}
	public void setCentroid(ArrayList<Double> cen){
		centroid=new ArrayList<Double>(cen);
	}
	public double getSE(){
		return se;
	}
	public void setSE(double serror){
		se=serror;
	}
	public void updateCentroid(int dim){
		if(data.size()==0){
			return;
		}
		int i,j;
		ArrayList<Double> newCen=new ArrayList<Double>();
		for(i=0;i<dim;i++){
			newCen.add(0.0);
		}
		int size=data.size();
		
		for(i=0;i<size;i++){
			ArrayList<Double> temp=data.get(i);
			for(j=0;j<dim;j++){
				newCen.set(j, (newCen.get(j)+temp.get(j)));
			}
		}
		for(j=0;j<dim;j++){
			newCen.set(j, (newCen.get(j)/size));
		}
		centroid=new ArrayList<Double>(newCen);
	}
}